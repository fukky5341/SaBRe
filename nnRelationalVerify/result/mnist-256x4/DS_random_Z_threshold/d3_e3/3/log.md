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
execution time: IAR + RelationalAnalysis = 0.87 + 2.61 = 3.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0023286, upper bound: 0.0023287

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0023167, upper bound: 0.0022871
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0022872, upper bound: 0.0023167
time: 1.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.80
Output dim: 5, lower bound: -0.0023167, upper bound: 0.0022871
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.80
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0023036, upper bound: 0.0022710
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0023004, upper bound: 0.0022741
time: 1.84 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091
time: 1.82 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.51 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 5, lower bound: -0.0023036, upper bound: 0.0022710
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 5, lower bound: -0.0023004, upper bound: 0.0022741
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 5, lower bound: -0.0021793, upper bound: 0.0022091

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076966, 0.0077301
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021699, 0.0021794
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160104, 0.0160802
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021187, 0.0021280
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120173, 0.0119652
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033388, 0.0033243
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030306, 0.0030174
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113096, 0.0112606
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087641, 0.0088023
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007594, 0.0007561

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019791, upper bound: 0.0019424
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019791, upper bound: 0.0019424
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077250, 0.0077017
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021780, 0.0021714
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160696, 0.0160210
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021265, 0.0021201
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119731, 0.0120094
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033265, 0.0033366
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030194, 0.0030286
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112680, 0.0113022
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087965, 0.0087699
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007566, 0.0007589

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021955, upper bound: 0.0021675
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021955, upper bound: 0.0021675
time: 1.67 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020415, upper bound: 0.0020689
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020404, upper bound: 0.0020697
time: 1.68 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020415, upper bound: 0.0020688
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020404, upper bound: 0.0020697
time: 1.64 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.12 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0019791, upper bound: 0.0019424
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0019791, upper bound: 0.0019424
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0021955, upper bound: 0.0021675
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0021955, upper bound: 0.0021675
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0020415, upper bound: 0.0020689
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0020404, upper bound: 0.0020697
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0020415, upper bound: 0.0020688
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 5, lower bound: -0.0020404, upper bound: 0.0020697

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076963, 0.0076918
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021699, 0.0021686
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160100, 0.0160005
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021187, 0.0021174
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119578, 0.0119649
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033222, 0.0033242
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030156, 0.0030174
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112536, 0.0112603
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087639, 0.0087587
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007557, 0.0007561

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021627, upper bound: 0.0021220
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021414, upper bound: 0.0021318
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077250, 0.0076730
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021780, 0.0021633
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160696, 0.0159614
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021265, 0.0021122
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119286, 0.0120094
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033141, 0.0033366
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030082, 0.0030286
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112261, 0.0113022
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087965, 0.0087373
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007538, 0.0007589

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021878, upper bound: 0.0021575
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021847, upper bound: 0.0021598
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077361, 0.0077892
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021811, 0.0021961
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160926, 0.0162031
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021296, 0.0021442
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0121092, 0.0120266
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033643, 0.0033414
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030538, 0.0030329
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113961, 0.0113184
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088091, 0.0088696
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007652, 0.0007600

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020055, upper bound: 0.0020182
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019971, upper bound: 0.0020353
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078157, 0.0077509
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022035, 0.0021853
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162581, 0.0161234
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021515, 0.0021337
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120496, 0.0121503
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033477, 0.0033757
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030387, 0.0030641
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113401, 0.0114348
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088997, 0.0088260
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007615, 0.0007678

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020261, upper bound: 0.0020541
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020234, upper bound: 0.0020554
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077637, 0.0077643
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021889, 0.0021890
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161502, 0.0161513
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021372, 0.0021374
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120705, 0.0120696
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033535, 0.0033533
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030440, 0.0030438
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113597, 0.0113589
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088406, 0.0088413
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007628, 0.0007627

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020240, upper bound: 0.0020443
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020157, upper bound: 0.0020517
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078436, 0.0077310
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022114, 0.0021796
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0163163, 0.0160820
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021592, 0.0021282
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120187, 0.0121938
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033391, 0.0033878
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030309, 0.0030751
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113109, 0.0114757
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089315, 0.0088033
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007595, 0.0007706

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020271, upper bound: 0.0020555
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020254, upper bound: 0.0020566
time: 1.72 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.11 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0021627, upper bound: 0.0021220
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0021414, upper bound: 0.0021318
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0021878, upper bound: 0.0021575
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0021847, upper bound: 0.0021598
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0020055, upper bound: 0.0020182
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0019971, upper bound: 0.0020353
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0020261, upper bound: 0.0020541
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0020234, upper bound: 0.0020554
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0020240, upper bound: 0.0020443
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0020157, upper bound: 0.0020517
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0020271, upper bound: 0.0020555
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 5, lower bound: -0.0020254, upper bound: 0.0020566

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075279, 0.0075502
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021224, 0.0021287
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156596, 0.0157059
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020723, 0.0020784
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117376, 0.0117030
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032611, 0.0032514
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029601, 0.0029513
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110464, 0.0110138
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085721, 0.0085974
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007417, 0.0007396

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021324, upper bound: 0.0020834
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021226, upper bound: 0.0020924
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075586, 0.0075234
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021311, 0.0021211
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157235, 0.0156502
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020808, 0.0020711
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116960, 0.0117508
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032495, 0.0032647
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029496, 0.0029634
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110072, 0.0110588
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086071, 0.0085669
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007391, 0.0007426

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021397, upper bound: 0.0020912
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020997, upper bound: 0.0021303
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076990, 0.0076500
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021706, 0.0021568
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160155, 0.0159135
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021194, 0.0021059
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118927, 0.0119690
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033042, 0.0033253
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029992, 0.0030184
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111924, 0.0112641
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087669, 0.0087111
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007515, 0.0007564

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021614, upper bound: 0.0021159
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021450, upper bound: 0.0021317
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077022, 0.0076463
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021715, 0.0021558
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160221, 0.0159058
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021203, 0.0021049
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118870, 0.0119739
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033026, 0.0033267
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029977, 0.0030196
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111870, 0.0112688
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087705, 0.0087069
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007512, 0.0007567

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021557, upper bound: 0.0020633
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020856, upper bound: 0.0021307
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075911, 0.0076734
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021402, 0.0021634
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157910, 0.0159622
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020897, 0.0021123
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119292, 0.0118012
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033143, 0.0032787
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030084, 0.0029761
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112267, 0.0111062
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086440, 0.0087377
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007538, 0.0007458

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019844, upper bound: 0.0019865
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019703, upper bound: 0.0019965
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076219, 0.0076442
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021489, 0.0021552
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158551, 0.0159014
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020982, 0.0021043
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118837, 0.0118491
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033017, 0.0032920
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029969, 0.0029882
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111839, 0.0111513
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086791, 0.0087045
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007510, 0.0007488

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0020248
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019869, upper bound: 0.0020249
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077885, 0.0077225
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021959, 0.0021773
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162017, 0.0160644
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021440, 0.0021259
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120055, 0.0121081
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033355, 0.0033640
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030276, 0.0030535
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112985, 0.0113951
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088688, 0.0087937
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007587, 0.0007652

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019208, upper bound: 0.0019456
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019208, upper bound: 0.0019456
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077878, 0.0077227
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021957, 0.0021773
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162003, 0.0160648
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021439, 0.0021259
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120058, 0.0121071
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033356, 0.0033637
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030277, 0.0030532
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112988, 0.0113941
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088681, 0.0087939
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007587, 0.0007651

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020131, upper bound: 0.0020451
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020132, upper bound: 0.0020451
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076250, 0.0076723
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021498, 0.0021631
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158615, 0.0159599
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020990, 0.0021120
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119275, 0.0118539
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033138, 0.0032934
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030079, 0.0029894
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112251, 0.0111558
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086826, 0.0087365
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007537, 0.0007491

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020081, upper bound: 0.0020184
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019960, upper bound: 0.0020282
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076663, 0.0076267
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021614, 0.0021503
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159475, 0.0158651
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021104, 0.0020995
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118566, 0.0119181
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032941, 0.0033112
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029901, 0.0030056
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111584, 0.0112163
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087297, 0.0086846
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007493, 0.0007532

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020014, upper bound: 0.0020359
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019977, upper bound: 0.0020373
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077722, 0.0076614
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021913, 0.0021600
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161677, 0.0159373
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021395, 0.0021091
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119106, 0.0120827
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033091, 0.0033569
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030037, 0.0030471
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112092, 0.0113712
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088502, 0.0087241
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007527, 0.0007636

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019181, upper bound: 0.0019487
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019181, upper bound: 0.0019487
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077788, 0.0076548
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021931, 0.0021582
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161814, 0.0159235
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021414, 0.0021072
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119002, 0.0120930
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033062, 0.0033598
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030011, 0.0030497
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111994, 0.0113809
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088577, 0.0087165
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007520, 0.0007642

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020042, upper bound: 0.0020220
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019927, upper bound: 0.0020361
time: 1.76 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.41 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0021324, upper bound: 0.0020834
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0021226, upper bound: 0.0020924
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0021397, upper bound: 0.0020912
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0020997, upper bound: 0.0021303
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0021614, upper bound: 0.0021159
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0021450, upper bound: 0.0021317
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0021557, upper bound: 0.0020633
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0020856, upper bound: 0.0021307
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019844, upper bound: 0.0019865
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019703, upper bound: 0.0019965
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0020248
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019869, upper bound: 0.0020249
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019208, upper bound: 0.0019456
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019208, upper bound: 0.0019456
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0020131, upper bound: 0.0020451
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0020132, upper bound: 0.0020451
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0020081, upper bound: 0.0020184
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019960, upper bound: 0.0020282
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0020014, upper bound: 0.0020359
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019977, upper bound: 0.0020373
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019181, upper bound: 0.0019487
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019181, upper bound: 0.0019487
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0020042, upper bound: 0.0020220
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 5, lower bound: -0.0019927, upper bound: 0.0020361

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074360, 0.0074732
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020965, 0.0021070
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154685, 0.0155457
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020470, 0.0020572
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116179, 0.0115602
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032278, 0.0032118
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029299, 0.0029153
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109337, 0.0108794
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084675, 0.0085097
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007342, 0.0007305

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020452, upper bound: 0.0019980
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020452, upper bound: 0.0019980
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074509, 0.0074539
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021007, 0.0021015
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154994, 0.0155057
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020511, 0.0020519
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115880, 0.0115833
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032195, 0.0032182
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029223, 0.0029211
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109056, 0.0109012
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084844, 0.0084878
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007323, 0.0007320

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015248, upper bound: 0.0015152
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015248, upper bound: 0.0015152
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075498, 0.0075368
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021286, 0.0021249
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157051, 0.0156780
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020783, 0.0020747
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117168, 0.0117370
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032553, 0.0032609
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029548, 0.0029599
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110268, 0.0110458
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085970, 0.0085822
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007404, 0.0007417

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020071, upper bound: 0.0019614
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020062, upper bound: 0.0019623
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075729, 0.0075145
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021351, 0.0021186
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157532, 0.0156317
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020847, 0.0020686
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116822, 0.0117729
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032457, 0.0032709
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029461, 0.0029690
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109942, 0.0110796
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086233, 0.0085568
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007382, 0.0007440

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019887, upper bound: 0.0019991
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019887, upper bound: 0.0019990
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075946, 0.0075836
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021412, 0.0021381
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157983, 0.0157755
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020907, 0.0020876
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117896, 0.0118067
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032755, 0.0032802
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029732, 0.0029775
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110953, 0.0111114
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086480, 0.0086355
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007450, 0.0007461

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021475, upper bound: 0.0020989
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021465, upper bound: 0.0021022
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076343, 0.0075463
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021524, 0.0021276
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158809, 0.0156978
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021016, 0.0020774
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117316, 0.0118684
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032594, 0.0032974
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029585, 0.0029930
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110407, 0.0111695
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086932, 0.0085930
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007414, 0.0007500

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020666, upper bound: 0.0020492
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020666, upper bound: 0.0020492
time: 1.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076888, 0.0077807
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021678, 0.0021937
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159943, 0.0161855
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021166, 0.0021419
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0120960, 0.0119532
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033606, 0.0033209
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030504, 0.0030144
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0113837, 0.0112493
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087553, 0.0088600
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007644, 0.0007554

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021318, upper bound: 0.0020008
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021084, upper bound: 0.0020402
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078320, 0.0076349
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022081, 0.0021526
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162921, 0.0158822
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021560, 0.0021018
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118694, 0.0121757
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032977, 0.0033828
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029933, 0.0030705
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111704, 0.0114587
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0089183, 0.0086939
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007501, 0.0007694

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019675, upper bound: 0.0019964
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019656, upper bound: 0.0019979
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075052, 0.0076286
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021160, 0.0021508
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156123, 0.0158691
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020660, 0.0021000
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118596, 0.0116677
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032949, 0.0032416
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029908, 0.0029424
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111612, 0.0109806
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085462, 0.0086867
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007495, 0.0007373

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019830, upper bound: 0.0019508
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019528, upper bound: 0.0019850
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075448, 0.0075875
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021272, 0.0021392
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156948, 0.0157835
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020770, 0.0020887
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117956, 0.0117293
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032772, 0.0032587
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029747, 0.0029580
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111010, 0.0110386
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085913, 0.0086399
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007454, 0.0007412

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019409, upper bound: 0.0019097
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018908, upper bound: 0.0019671
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074485, 0.0075009
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021000, 0.0021148
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154945, 0.0156034
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020504, 0.0020649
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116610, 0.0115796
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032398, 0.0032172
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029407, 0.0029202
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109743, 0.0108977
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084817, 0.0085413
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007369, 0.0007318

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019702, upper bound: 0.0019979
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019609, upper bound: 0.0020089
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074786, 0.0074677
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021085, 0.0021054
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155570, 0.0155344
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020587, 0.0020557
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116094, 0.0116264
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032254, 0.0032301
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029277, 0.0029320
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109258, 0.0109417
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085159, 0.0085036
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007336, 0.0007347

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019792, upper bound: 0.0020174
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019771, upper bound: 0.0020174
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076602, 0.0076173
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021597, 0.0021476
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159348, 0.0158455
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021087, 0.0020969
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118419, 0.0119087
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032900, 0.0033086
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029864, 0.0030032
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111446, 0.0112074
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087227, 0.0086738
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007483, 0.0007526

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019851, upper bound: 0.0020109
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019794, upper bound: 0.0020170
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076878, 0.0075875
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021675, 0.0021392
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159921, 0.0157836
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021163, 0.0020887
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117957, 0.0119515
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032772, 0.0033205
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029747, 0.0030140
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111010, 0.0112477
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087541, 0.0086400
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007454, 0.0007553

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019073, upper bound: 0.0019365
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019073, upper bound: 0.0019365
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075635, 0.0076402
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021324, 0.0021541
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157337, 0.0158931
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020821, 0.0021032
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118775, 0.0117584
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032999, 0.0032668
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029953, 0.0029653
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111781, 0.0110660
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086126, 0.0086999
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007506, 0.0007431

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019733, upper bound: 0.0019695
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019634, upper bound: 0.0019847
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075917, 0.0076111
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021404, 0.0021459
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157922, 0.0158327
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020898, 0.0020952
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118324, 0.0118021
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032874, 0.0032790
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029840, 0.0029763
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111356, 0.0111071
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086447, 0.0086668
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007477, 0.0007458

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019758, upper bound: 0.0019920
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019643, upper bound: 0.0020080
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076383, 0.0075984
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021535, 0.0021423
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158891, 0.0158062
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021027, 0.0020917
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118125, 0.0118746
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032819, 0.0032991
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029789, 0.0029946
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111169, 0.0111753
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086977, 0.0086523
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007465, 0.0007504

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019814, upper bound: 0.0020015
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019673, upper bound: 0.0020153
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076383, 0.0075985
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021535, 0.0021423
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158892, 0.0158065
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021027, 0.0020917
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118128, 0.0118746
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032819, 0.0032991
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029790, 0.0029946
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111172, 0.0111753
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086978, 0.0086525
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007465, 0.0007504

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015784, upper bound: 0.0015970
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015784, upper bound: 0.0015970
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076842, 0.0076027
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021665, 0.0021435
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0159848, 0.0158151
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021153, 0.0020929
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118192, 0.0119460
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032837, 0.0033190
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029806, 0.0030126
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111232, 0.0112425
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087501, 0.0086572
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007469, 0.0007549

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019794, upper bound: 0.0019974
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019779, upper bound: 0.0019977
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077239, 0.0075633
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021776, 0.0021324
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0160672, 0.0157332
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021262, 0.0020820
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117580, 0.0120076
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032667, 0.0033361
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029652, 0.0030282
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110656, 0.0113005
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0087952, 0.0086124
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007430, 0.0007588

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019106, upper bound: 0.0019337
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018841, upper bound: 0.0019522
time: 1.99 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.60 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0020452, upper bound: 0.0019980
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0020452, upper bound: 0.0019980
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0015248, upper bound: 0.0015152
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0015248, upper bound: 0.0015152
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0020071, upper bound: 0.0019614
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0020062, upper bound: 0.0019623
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019887, upper bound: 0.0019991
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019887, upper bound: 0.0019990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0021475, upper bound: 0.0020989
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0021465, upper bound: 0.0021022
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0020666, upper bound: 0.0020492
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0020666, upper bound: 0.0020492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0021318, upper bound: 0.0020008
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0021084, upper bound: 0.0020402
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019675, upper bound: 0.0019964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019656, upper bound: 0.0019979
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019830, upper bound: 0.0019508
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019528, upper bound: 0.0019850
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019409, upper bound: 0.0019097
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0018908, upper bound: 0.0019671
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019702, upper bound: 0.0019979
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019609, upper bound: 0.0020089
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019792, upper bound: 0.0020174
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019771, upper bound: 0.0020174
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019851, upper bound: 0.0020109
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019794, upper bound: 0.0020170
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019073, upper bound: 0.0019365
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019073, upper bound: 0.0019365
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019733, upper bound: 0.0019695
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019634, upper bound: 0.0019847
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019758, upper bound: 0.0019920
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019643, upper bound: 0.0020080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019814, upper bound: 0.0020015
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019673, upper bound: 0.0020153
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0015784, upper bound: 0.0015970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0015784, upper bound: 0.0015970
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019794, upper bound: 0.0019974
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019779, upper bound: 0.0019977
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0019106, upper bound: 0.0019337
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 5, lower bound: -0.0018841, upper bound: 0.0019522

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073351, 0.0073990
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020680, 0.0020861
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0152585, 0.0153914
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020192, 0.0020368
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115026, 0.0114032
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031958, 0.0031682
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029008, 0.0028757
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108252, 0.0107317
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083525, 0.0084253
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007269, 0.0007206

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020337, upper bound: 0.0019767
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020194, upper bound: 0.0019865
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074360, 0.0073722
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020965, 0.0020785
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154685, 0.0153357
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020470, 0.0020294
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114609, 0.0115602
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031842, 0.0032118
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028903, 0.0029153
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107860, 0.0108794
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084675, 0.0083948
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007243, 0.0007305

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020434, upper bound: 0.0019702
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020073, upper bound: 0.0019961
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074652, 0.0074875
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021047, 0.0021110
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155292, 0.0155756
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020550, 0.0020612
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116402, 0.0116056
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032340, 0.0032244
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029355, 0.0029268
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109548, 0.0109221
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085007, 0.0085261
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007356, 0.0007334

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019928, upper bound: 0.0019432
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019910, upper bound: 0.0019472
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075498, 0.0074522
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021286, 0.0021011
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157051, 0.0155022
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020783, 0.0020515
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115853, 0.0117370
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032188, 0.0032609
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029217, 0.0029599
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109031, 0.0110458
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085970, 0.0084859
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007321, 0.0007417

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019919, upper bound: 0.0019437
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0019483
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075398, 0.0074872
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021258, 0.0021109
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156843, 0.0155750
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020756, 0.0020611
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116397, 0.0117215
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032339, 0.0032566
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029354, 0.0029560
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109543, 0.0110312
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085856, 0.0085257
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007356, 0.0007407

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016010
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016010
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075729, 0.0074814
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021351, 0.0021093
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157532, 0.0155629
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020847, 0.0020595
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116307, 0.0117729
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032314, 0.0032709
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029331, 0.0029690
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109458, 0.0110796
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086233, 0.0085192
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007350, 0.0007440

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019690, upper bound: 0.0019684
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019590, upper bound: 0.0019790
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075798, 0.0075695
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021370, 0.0021341
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157676, 0.0157461
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020866, 0.0020837
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117676, 0.0117837
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032694, 0.0032739
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029676, 0.0029717
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110747, 0.0110898
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086312, 0.0086194
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007436, 0.0007447

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021243, upper bound: 0.0020483
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021007, upper bound: 0.0020754
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075790, 0.0075691
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021368, 0.0021340
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157658, 0.0157452
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020864, 0.0020836
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117670, 0.0117824
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032692, 0.0032735
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029675, 0.0029713
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110740, 0.0110885
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086302, 0.0086189
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007436, 0.0007446

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015337, upper bound: 0.0015256
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015337, upper bound: 0.0015256
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075944, 0.0075071
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021411, 0.0021165
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157978, 0.0156163
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020906, 0.0020666
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116706, 0.0118063
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032425, 0.0032801
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029432, 0.0029774
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109834, 0.0111111
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086478, 0.0085484
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007375, 0.0007461

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020042
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019988, upper bound: 0.0020129
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075951, 0.0075060
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021413, 0.0021162
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157994, 0.0156140
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020908, 0.0020663
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116690, 0.0118075
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032420, 0.0032805
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029427, 0.0029777
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109818, 0.0111122
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086486, 0.0085471
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007374, 0.0007462

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020378, upper bound: 0.0020130
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020201
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075468, 0.0076873
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021277, 0.0021673
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156989, 0.0159912
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020775, 0.0021162
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119508, 0.0117324
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033203, 0.0032596
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030138, 0.0029587
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112470, 0.0110415
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085936, 0.0087536
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007552, 0.0007414

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021177, upper bound: 0.0019810
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0021166, upper bound: 0.0019870
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076000, 0.0076391
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021427, 0.0021538
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158096, 0.0158909
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020921, 0.0021029
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118759, 0.0118151
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032995, 0.0032826
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029949, 0.0029796
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111765, 0.0111193
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086542, 0.0086987
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007505, 0.0007466

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020944, upper bound: 0.0020241
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020927, upper bound: 0.0020261
time: 1.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0077496, 0.0076158
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021849, 0.0021472
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0161207, 0.0158424
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021333, 0.0020965
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118396, 0.0120476
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032894, 0.0033472
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029858, 0.0030382
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111424, 0.0113381
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088245, 0.0086721
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007482, 0.0007613

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019328, upper bound: 0.0019518
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019216, upper bound: 0.0019624
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0078132, 0.0075522
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0022028, 0.0021292
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0162530, 0.0157101
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021508, 0.0020790
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117407, 0.0121465
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032619, 0.0033747
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029608, 0.0030632
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110493, 0.0114312
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0088969, 0.0085997
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007419, 0.0007676

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018171, upper bound: 0.0018400
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018171, upper bound: 0.0018400
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075096, 0.0076566
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021172, 0.0021587
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156214, 0.0159272
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020672, 0.0021077
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119030, 0.0116745
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033070, 0.0032435
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030018, 0.0029441
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112021, 0.0109870
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085512, 0.0087186
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007522, 0.0007378

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019582, upper bound: 0.0019251
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019567, upper bound: 0.0019263
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075309, 0.0076330
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021232, 0.0021520
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156657, 0.0158782
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020731, 0.0021012
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118664, 0.0117076
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032968, 0.0032527
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029925, 0.0029525
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111676, 0.0110181
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085754, 0.0086917
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007499, 0.0007398

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019312, upper bound: 0.0019459
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019118, upper bound: 0.0019629
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073907, 0.0074748
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020837, 0.0021074
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153741, 0.0155491
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020345, 0.0020577
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116204, 0.0114897
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032285, 0.0031922
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029305, 0.0028975
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109361, 0.0108130
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084158, 0.0085116
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007343, 0.0007261

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019488, upper bound: 0.0019629
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019358, upper bound: 0.0019773
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074179, 0.0074430
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020914, 0.0020985
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154308, 0.0154831
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020420, 0.0020489
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115711, 0.0115320
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032148, 0.0032039
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029181, 0.0029082
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108897, 0.0108529
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084468, 0.0084754
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007312, 0.0007288

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018794, upper bound: 0.0019059
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018542, upper bound: 0.0019263
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074529, 0.0074458
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021012, 0.0020993
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155035, 0.0154889
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020516, 0.0020497
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115754, 0.0115863
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032160, 0.0032190
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029191, 0.0029219
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108937, 0.0109040
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084866, 0.0084786
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007315, 0.0007322

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0019266
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018950, upper bound: 0.0019886
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074567, 0.0074417
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021023, 0.0020981
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155115, 0.0154802
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020527, 0.0020486
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115689, 0.0115923
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032142, 0.0032207
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029175, 0.0029234
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108876, 0.0109097
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084910, 0.0084739
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007311, 0.0007326

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019471, upper bound: 0.0019262
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0018950, upper bound: 0.0019887
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075591, 0.0075318
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021312, 0.0021235
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157245, 0.0156677
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020809, 0.0020734
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117091, 0.0117515
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032531, 0.0032649
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029529, 0.0029636
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110195, 0.0110595
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086076, 0.0085765
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007399, 0.0007426

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019804, upper bound: 0.0019965
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019687, upper bound: 0.0020061
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075778, 0.0075154
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021365, 0.0021189
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157633, 0.0156335
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020860, 0.0020688
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116835, 0.0117805
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032460, 0.0032730
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029464, 0.0029709
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109955, 0.0110868
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086288, 0.0085578
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007383, 0.0007445

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019634, upper bound: 0.0019899
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019526, upper bound: 0.0020010
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074468, 0.0074934
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020995, 0.0021127
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154908, 0.0155878
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020500, 0.0020628
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116494, 0.0115769
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032365, 0.0032164
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029378, 0.0029195
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109633, 0.0108951
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084797, 0.0085328
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007362, 0.0007316

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019426, upper bound: 0.0019475
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019311, upper bound: 0.0019648
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075009, 0.0075600
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021148, 0.0021314
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156033, 0.0157263
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020648, 0.0020811
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117529, 0.0116609
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032653, 0.0032398
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029639, 0.0029407
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110607, 0.0109742
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085413, 0.0086086
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007427, 0.0007369

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019684, upper bound: 0.0019841
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019678, upper bound: 0.0019844
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075440, 0.0075204
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021269, 0.0021203
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156931, 0.0156440
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020767, 0.0020702
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116913, 0.0117281
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032482, 0.0032584
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029484, 0.0029576
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110028, 0.0110374
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085904, 0.0085635
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007388, 0.0007411

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018825, upper bound: 0.0019060
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018582, upper bound: 0.0019237
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075450, 0.0075484
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021272, 0.0021282
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156951, 0.0157022
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020770, 0.0020779
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117348, 0.0117295
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032603, 0.0032588
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029594, 0.0029580
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110438, 0.0110388
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085915, 0.0085954
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007416, 0.0007412

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019652, upper bound: 0.0019744
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019557, upper bound: 0.0019855
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075865, 0.0075053
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021389, 0.0021160
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157814, 0.0156125
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020884, 0.0020661
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116678, 0.0117941
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032417, 0.0032767
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029424, 0.0029743
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109807, 0.0110995
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086388, 0.0085463
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007373, 0.0007453

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019626, upper bound: 0.0019986
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019508, upper bound: 0.0020105
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076136, 0.0075231
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021466, 0.0021211
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158379, 0.0156497
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020959, 0.0020710
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116956, 0.0118362
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032494, 0.0032885
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029495, 0.0029849
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110069, 0.0111392
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086697, 0.0085667
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007391, 0.0007480

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019512, upper bound: 0.0019640
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019457, upper bound: 0.0019694
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076084, 0.0075277
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021451, 0.0021224
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158271, 0.0156592
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020945, 0.0020722
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117027, 0.0118282
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032514, 0.0032862
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029513, 0.0029829
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110136, 0.0111316
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086638, 0.0085719
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007395, 0.0007475

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019732, upper bound: 0.0019854
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019614, upper bound: 0.0019929
time: 1.90 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.73 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020337, upper bound: 0.0019767
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020194, upper bound: 0.0019865
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020434, upper bound: 0.0019702
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020073, upper bound: 0.0019961
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019928, upper bound: 0.0019432
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019910, upper bound: 0.0019472
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019919, upper bound: 0.0019437
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019902, upper bound: 0.0019483
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019690, upper bound: 0.0019684
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019590, upper bound: 0.0019790
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0021243, upper bound: 0.0020483
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0021007, upper bound: 0.0020754
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0015337, upper bound: 0.0015256
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0015337, upper bound: 0.0015256
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020042
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019988, upper bound: 0.0020129
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020378, upper bound: 0.0020130
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020299, upper bound: 0.0020201
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0021177, upper bound: 0.0019810
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0021166, upper bound: 0.0019870
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020944, upper bound: 0.0020241
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0020927, upper bound: 0.0020261
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019328, upper bound: 0.0019518
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019216, upper bound: 0.0019624
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018171, upper bound: 0.0018400
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018171, upper bound: 0.0018400
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019582, upper bound: 0.0019251
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019567, upper bound: 0.0019263
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019312, upper bound: 0.0019459
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019118, upper bound: 0.0019629
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019488, upper bound: 0.0019629
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019358, upper bound: 0.0019773
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018794, upper bound: 0.0019059
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018542, upper bound: 0.0019263
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019496, upper bound: 0.0019266
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018950, upper bound: 0.0019886
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019471, upper bound: 0.0019262
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018950, upper bound: 0.0019887
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019804, upper bound: 0.0019965
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019687, upper bound: 0.0020061
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019634, upper bound: 0.0019899
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019526, upper bound: 0.0020010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019426, upper bound: 0.0019475
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019311, upper bound: 0.0019648
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019684, upper bound: 0.0019841
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019678, upper bound: 0.0019844
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018825, upper bound: 0.0019060
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0018582, upper bound: 0.0019237
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019652, upper bound: 0.0019744
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019557, upper bound: 0.0019855
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019626, upper bound: 0.0019986
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019508, upper bound: 0.0020105
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019512, upper bound: 0.0019640
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019457, upper bound: 0.0019694
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019732, upper bound: 0.0019854
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.73
Output dim: 5, lower bound: -0.0019614, upper bound: 0.0019929

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073025, 0.0073755
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020588, 0.0020794
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0151906, 0.0153424
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020102, 0.0020303
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114660, 0.0113525
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031856, 0.0031541
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028916, 0.0028629
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107908, 0.0106840
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083154, 0.0083985
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007246, 0.0007174

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020088, upper bound: 0.0019505
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020085, upper bound: 0.0019515
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073116, 0.0073663
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020614, 0.0020768
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0152095, 0.0153235
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020127, 0.0020278
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114518, 0.0113666
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031817, 0.0031580
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028880, 0.0028665
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107774, 0.0106973
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083257, 0.0083881
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007237, 0.0007183

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019943, upper bound: 0.0019599
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019933, upper bound: 0.0019613
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074218, 0.0073819
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020925, 0.0020812
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154389, 0.0153558
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020431, 0.0020321
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114760, 0.0115381
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031884, 0.0032056
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028941, 0.0029097
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108002, 0.0108586
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084513, 0.0084058
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007252, 0.0007291

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018886, upper bound: 0.0018236
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018884, upper bound: 0.0018236
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074427, 0.0073591
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020984, 0.0020748
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154824, 0.0153084
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020489, 0.0020258
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114406, 0.0115706
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031785, 0.0032147
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028851, 0.0029179
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107669, 0.0108892
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084751, 0.0083799
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007230, 0.0007312

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019836, upper bound: 0.0019536
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019618, upper bound: 0.0019726
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074515, 0.0074744
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021009, 0.0021073
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155006, 0.0155483
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020513, 0.0020576
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116198, 0.0115842
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032283, 0.0032184
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029304, 0.0029214
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109356, 0.0109020
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084851, 0.0085112
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007343, 0.0007320

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019767, upper bound: 0.0019206
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019663, upper bound: 0.0019269
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074512, 0.0074738
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021008, 0.0021071
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155000, 0.0155470
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020512, 0.0020574
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116189, 0.0115838
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032281, 0.0032183
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029301, 0.0029213
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109347, 0.0109016
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084847, 0.0085105
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007342, 0.0007320

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015548, upper bound: 0.0015440
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015548, upper bound: 0.0015440
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075366, 0.0074388
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021248, 0.0020973
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156776, 0.0154741
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020747, 0.0020478
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115644, 0.0117165
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032129, 0.0032552
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029164, 0.0029547
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108834, 0.0110265
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085819, 0.0084706
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007308, 0.0007404

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019640, upper bound: 0.0019143
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0019164
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075363, 0.0074385
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021248, 0.0020972
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156770, 0.0154736
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020746, 0.0020477
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115640, 0.0117160
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032128, 0.0032551
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029163, 0.0029546
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108830, 0.0110261
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085816, 0.0084703
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007308, 0.0007404

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019606, upper bound: 0.0018849
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0019172
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074708, 0.0075195
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021063, 0.0021200
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155408, 0.0156421
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020566, 0.0020700
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116899, 0.0116142
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032478, 0.0032268
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029480, 0.0029289
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110015, 0.0109303
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085070, 0.0085625
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007387, 0.0007339

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020415, upper bound: 0.0019495
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020193, upper bound: 0.0019590
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075243, 0.0074607
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021214, 0.0021034
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156521, 0.0155198
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020713, 0.0020538
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115985, 0.0116974
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032224, 0.0032499
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029250, 0.0029499
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109155, 0.0110086
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085680, 0.0084955
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007330, 0.0007392

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019550, upper bound: 0.0019382
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019550, upper bound: 0.0019382
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074300, 0.0073717
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020948, 0.0020784
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154559, 0.0153346
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020453, 0.0020293
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114601, 0.0115508
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031840, 0.0032092
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028901, 0.0029129
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107853, 0.0108706
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084606, 0.0083942
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007242, 0.0007299

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018721, upper bound: 0.0018384
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018718, upper bound: 0.0018386
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074619, 0.0073425
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021038, 0.0020701
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155222, 0.0152739
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020541, 0.0020213
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114147, 0.0116003
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031714, 0.0032229
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028786, 0.0029254
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107425, 0.0109172
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084969, 0.0083609
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007213, 0.0007331

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019351, upper bound: 0.0019092
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019124, upper bound: 0.0019273
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075064, 0.0074302
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021163, 0.0020948
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156149, 0.0154562
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020664, 0.0020454
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115510, 0.0116696
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032092, 0.0032422
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029130, 0.0029429
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108708, 0.0109824
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085476, 0.0084608
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007300, 0.0007374

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020010, upper bound: 0.0019668
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019899, upper bound: 0.0019764
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075206, 0.0074134
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021204, 0.0020901
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156445, 0.0154215
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020703, 0.0020408
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115251, 0.0116917
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032020, 0.0032483
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029065, 0.0029485
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108464, 0.0110032
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085638, 0.0084417
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007283, 0.0007388

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020005, upper bound: 0.0019313
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019475, upper bound: 0.0019915
time: 1.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075299, 0.0076700
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021230, 0.0021624
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156637, 0.0159551
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020728, 0.0021114
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119238, 0.0117061
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033128, 0.0032523
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030070, 0.0029521
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112216, 0.0110167
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085743, 0.0087338
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007535, 0.0007398

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020925, upper bound: 0.0019531
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020766, upper bound: 0.0019564
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075293, 0.0076702
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021228, 0.0021625
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156626, 0.0159556
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020727, 0.0021115
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0119242, 0.0117052
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0033129, 0.0032521
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0030071, 0.0029519
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0112220, 0.0110159
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085737, 0.0087341
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007535, 0.0007397

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013027, upper bound: 0.0012937
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013027, upper bound: 0.0012937
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075831, 0.0076216
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021380, 0.0021488
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157744, 0.0158544
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020875, 0.0020981
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118486, 0.0117888
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032919, 0.0032753
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029880, 0.0029730
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111509, 0.0110946
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086349, 0.0086787
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007488, 0.0007450

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020697, upper bound: 0.0019911
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020548, upper bound: 0.0019989
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075823, 0.0076220
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021377, 0.0021489
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157727, 0.0158553
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020873, 0.0020982
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118493, 0.0117875
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032921, 0.0032749
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029882, 0.0029726
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111515, 0.0110934
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086340, 0.0086792
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007488, 0.0007449

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020771, upper bound: 0.0020037
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020612, upper bound: 0.0020103
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075967, 0.0074451
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021418, 0.0020990
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158027, 0.0154872
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020912, 0.0020495
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115742, 0.0118100
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032157, 0.0032812
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029188, 0.0029783
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108926, 0.0111145
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086504, 0.0084777
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007314, 0.0007463

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018785, upper bound: 0.0019612
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018736, upper bound: 0.0019727
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076005, 0.0074409
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021429, 0.0020979
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158106, 0.0154785
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020923, 0.0020483
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115677, 0.0118159
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032138, 0.0032828
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029172, 0.0029798
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108865, 0.0111200
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086547, 0.0084730
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007310, 0.0007467

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015019, upper bound: 0.0015438
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015019, upper bound: 0.0015438
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075264, 0.0075078
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021220, 0.0021167
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156565, 0.0156178
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020719, 0.0020668
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116717, 0.0117007
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032428, 0.0032508
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029434, 0.0029507
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109844, 0.0110117
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085704, 0.0085492
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007376, 0.0007394

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019645, upper bound: 0.0019721
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019524, upper bound: 0.0019805
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075351, 0.0074990
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021244, 0.0021142
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156746, 0.0155994
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020743, 0.0020643
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116580, 0.0117142
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032389, 0.0032545
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029400, 0.0029541
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109715, 0.0110244
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085803, 0.0085391
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007367, 0.0007403

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018866, upper bound: 0.0019042
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018627, upper bound: 0.0019212
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075113, 0.0074754
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021177, 0.0021076
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156251, 0.0155504
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020677, 0.0020578
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116214, 0.0116772
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032288, 0.0032443
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029307, 0.0029448
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109370, 0.0109896
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085532, 0.0085123
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007344, 0.0007379

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019332, upper bound: 0.0019073
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018778, upper bound: 0.0019601
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075405, 0.0074484
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021260, 0.0021000
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156859, 0.0154941
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020758, 0.0020504
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115793, 0.0117226
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032171, 0.0032569
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029201, 0.0029563
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108974, 0.0110323
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085865, 0.0084815
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007317, 0.0007408

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019479, upper bound: 0.0019833
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019407, upper bound: 0.0019962
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074765, 0.0075393
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021079, 0.0021256
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155526, 0.0156832
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020581, 0.0020754
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117207, 0.0116230
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032564, 0.0032292
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029558, 0.0029312
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110305, 0.0109386
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085135, 0.0085850
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007407, 0.0007345

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019542, upper bound: 0.0019678
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019502, upper bound: 0.0019700
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074806, 0.0075355
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021091, 0.0021245
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155612, 0.0156754
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020593, 0.0020744
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117148, 0.0116295
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032547, 0.0032310
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029543, 0.0029328
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110249, 0.0109446
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085182, 0.0085807
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007403, 0.0007349

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019321, upper bound: 0.0019375
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019258, upper bound: 0.0019494
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075154, 0.0074912
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021189, 0.0021121
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156335, 0.0155833
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020688, 0.0020622
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116460, 0.0116835
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032356, 0.0032460
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029370, 0.0029464
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109602, 0.0109954
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085578, 0.0085303
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007360, 0.0007383

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018564, upper bound: 0.0018838
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018564, upper bound: 0.0018838
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075513, 0.0074803
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021290, 0.0021090
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157083, 0.0155606
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020787, 0.0020592
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116290, 0.0117394
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032309, 0.0032616
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029327, 0.0029605
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109442, 0.0110481
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085988, 0.0085179
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007349, 0.0007419

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019409, upper bound: 0.0019653
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019253, upper bound: 0.0019767
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075615, 0.0074707
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021319, 0.0021063
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157295, 0.0155406
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020816, 0.0020566
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116141, 0.0117553
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032267, 0.0032660
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029289, 0.0029645
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109302, 0.0110630
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086104, 0.0085070
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007339, 0.0007429

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018724, upper bound: 0.0019081
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018487, upper bound: 0.0019277
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075756, 0.0075041
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021358, 0.0021157
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157588, 0.0156101
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020854, 0.0020657
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116660, 0.0117771
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032412, 0.0032720
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029420, 0.0029700
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109790, 0.0110836
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086264, 0.0085450
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007372, 0.0007442

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0019600
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019499, upper bound: 0.0019688
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075848, 0.0074952
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021384, 0.0021132
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157780, 0.0155915
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020880, 0.0020633
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0116521, 0.0117915
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032373, 0.0032760
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029385, 0.0029736
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0109660, 0.0110971
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086369, 0.0085348
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007363, 0.0007451

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019447, upper bound: 0.0019663
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019400, upper bound: 0.0019766
time: 1.81 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.71 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020088, upper bound: 0.0019505
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020085, upper bound: 0.0019515
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019943, upper bound: 0.0019599
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019933, upper bound: 0.0019613
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018886, upper bound: 0.0018236
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018884, upper bound: 0.0018236
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019836, upper bound: 0.0019536
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019618, upper bound: 0.0019726
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019767, upper bound: 0.0019206
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019663, upper bound: 0.0019269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0015548, upper bound: 0.0015440
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0015548, upper bound: 0.0015440
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019640, upper bound: 0.0019143
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0019164
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019606, upper bound: 0.0018849
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018828, upper bound: 0.0019172
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020415, upper bound: 0.0019495
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020193, upper bound: 0.0019590
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019550, upper bound: 0.0019382
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019550, upper bound: 0.0019382
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018721, upper bound: 0.0018384
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018718, upper bound: 0.0018386
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019351, upper bound: 0.0019092
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019124, upper bound: 0.0019273
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020010, upper bound: 0.0019668
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019899, upper bound: 0.0019764
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020005, upper bound: 0.0019313
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019475, upper bound: 0.0019915
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020925, upper bound: 0.0019531
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020766, upper bound: 0.0019564
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0013027, upper bound: 0.0012937
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0013027, upper bound: 0.0012937
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020697, upper bound: 0.0019911
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020548, upper bound: 0.0019989
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020771, upper bound: 0.0020037
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0020612, upper bound: 0.0020103
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018785, upper bound: 0.0019612
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018736, upper bound: 0.0019727
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0015019, upper bound: 0.0015438
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0015019, upper bound: 0.0015438
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019645, upper bound: 0.0019721
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019524, upper bound: 0.0019805
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018866, upper bound: 0.0019042
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018627, upper bound: 0.0019212
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019332, upper bound: 0.0019073
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018778, upper bound: 0.0019601
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019479, upper bound: 0.0019833
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019407, upper bound: 0.0019962
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019542, upper bound: 0.0019678
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019502, upper bound: 0.0019700
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019321, upper bound: 0.0019375
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019258, upper bound: 0.0019494
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018564, upper bound: 0.0018838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018564, upper bound: 0.0018838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019409, upper bound: 0.0019653
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019253, upper bound: 0.0019767
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018724, upper bound: 0.0019081
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0018487, upper bound: 0.0019277
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019566, upper bound: 0.0019600
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019499, upper bound: 0.0019688
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019447, upper bound: 0.0019663
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.71
Output dim: 5, lower bound: -0.0019400, upper bound: 0.0019766

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072306, 0.0072927
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020386, 0.0020561
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150411, 0.0151703
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019904, 0.0020075
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113373, 0.0112408
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031498, 0.0031230
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028591, 0.0028348
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106697, 0.0105788
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082335, 0.0083042
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007164, 0.0007103

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019817, upper bound: 0.0018679
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019157, upper bound: 0.0019223
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072197, 0.0072990
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020355, 0.0020579
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150184, 0.0151833
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019875, 0.0020093
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113471, 0.0112238
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031526, 0.0031183
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028616, 0.0028305
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106789, 0.0105629
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082211, 0.0083114
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007171, 0.0007093

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018610, upper bound: 0.0018089
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018608, upper bound: 0.0018089
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072389, 0.0072836
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020409, 0.0020535
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150584, 0.0151513
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019927, 0.0020050
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113232, 0.0112537
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031459, 0.0031266
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028555, 0.0028380
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106564, 0.0105910
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082430, 0.0082939
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007156, 0.0007112

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018507, upper bound: 0.0018163
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018507, upper bound: 0.0018163
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072288, 0.0072900
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020381, 0.0020553
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150374, 0.0151647
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019900, 0.0020068
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113331, 0.0112380
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031487, 0.0031222
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028581, 0.0028341
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106657, 0.0105762
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082315, 0.0083012
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007162, 0.0007102

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019694, upper bound: 0.0019197
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019511, upper bound: 0.0019381
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073217, 0.0072898
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020643, 0.0020553
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0152307, 0.0151642
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020155, 0.0020067
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113328, 0.0113825
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031486, 0.0031624
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028580, 0.0028705
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106654, 0.0107122
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083373, 0.0083009
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007162, 0.0007193

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019692, upper bound: 0.0019352
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019680, upper bound: 0.0019396
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072200, 0.0073435
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020356, 0.0020704
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150191, 0.0152759
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019875, 0.0020215
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114163, 0.0112243
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031718, 0.0031184
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028790, 0.0028306
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107440, 0.0105633
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082214, 0.0083621
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007214, 0.0007093

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020335, upper bound: 0.0019338
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020190, upper bound: 0.0019423
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072738, 0.0072708
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020508, 0.0020499
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0151310, 0.0151247
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020023, 0.0020015
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113033, 0.0113079
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031404, 0.0031417
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028505, 0.0028517
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106376, 0.0106420
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082827, 0.0082793
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007143, 0.0007146

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019900, upper bound: 0.0019267
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019823, upper bound: 0.0019306
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073429, 0.0072950
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020702, 0.0020567
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0152748, 0.0151752
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020214, 0.0020082
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113410, 0.0114154
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031509, 0.0031715
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028600, 0.0028788
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106731, 0.0107432
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083615, 0.0083069
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007167, 0.0007214

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019759, upper bound: 0.0019401
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019759, upper bound: 0.0019419
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0073760, 0.0072664
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020796, 0.0020487
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0153435, 0.0151156
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020305, 0.0020003
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0112965, 0.0114668
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031385, 0.0031858
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028488, 0.0028918
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106312, 0.0107915
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0083991, 0.0082743
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007139, 0.0007246

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019650, upper bound: 0.0019507
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019642, upper bound: 0.0019512
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074911, 0.0075297
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021120, 0.0021229
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155830, 0.0156633
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020622, 0.0020728
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117058, 0.0116458
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032522, 0.0032355
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029520, 0.0029369
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110164, 0.0109600
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085302, 0.0085741
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007397, 0.0007359

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019864, upper bound: 0.0019158
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0019175
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0076355, 0.0073860
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021527, 0.0020824
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0158833, 0.0153643
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0021019, 0.0020332
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0114823, 0.0118702
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031901, 0.0032979
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028957, 0.0029935
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108061, 0.0111712
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086946, 0.0084104
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007256, 0.0007501

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019105, upper bound: 0.0019457
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019007, upper bound: 0.0019555
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074339, 0.0076179
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020959, 0.0021478
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0154639, 0.0158468
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020464, 0.0020971
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118429, 0.0115568
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032903, 0.0032108
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029866, 0.0029145
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111455, 0.0108762
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0084650, 0.0086746
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007484, 0.0007303

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020628, upper bound: 0.0019246
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020549, upper bound: 0.0019260
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074735, 0.0075751
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021071, 0.0021357
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155463, 0.0157577
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020573, 0.0020853
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117763, 0.0116184
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032718, 0.0032279
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029698, 0.0029300
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110828, 0.0109342
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085101, 0.0086258
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007442, 0.0007342

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020599, upper bound: 0.0019099
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0020423, upper bound: 0.0019392
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074871, 0.0075672
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021109, 0.0021335
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155746, 0.0157414
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020611, 0.0020831
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117641, 0.0116395
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032684, 0.0032338
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029667, 0.0029353
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110714, 0.0109541
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085256, 0.0086169
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007434, 0.0007355

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014733, upper bound: 0.0014528
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014733, upper bound: 0.0014528
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075293, 0.0075267
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021228, 0.0021221
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156625, 0.0156570
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020727, 0.0020720
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117011, 0.0117052
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032509, 0.0032520
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029508, 0.0029519
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110120, 0.0110159
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085737, 0.0085707
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007394, 0.0007397

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014732, upper bound: 0.0014528
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014732, upper bound: 0.0014528
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075450, 0.0076163
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021272, 0.0021473
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156951, 0.0158434
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020770, 0.0020966
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0118404, 0.0117296
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032896, 0.0032588
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029860, 0.0029580
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0111431, 0.0110388
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085915, 0.0086727
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007482, 0.0007412

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019442, upper bound: 0.0018866
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019442, upper bound: 0.0018866
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075692, 0.0075861
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021340, 0.0021388
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0157455, 0.0157806
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020837, 0.0020883
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0117935, 0.0117672
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032766, 0.0032693
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029741, 0.0029675
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0110990, 0.0110742
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0086191, 0.0086383
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007453, 0.0007436

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017343, upper bound: 0.0017134
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017343, upper bound: 0.0017134
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0074869, 0.0074414
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021108, 0.0020980
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0155743, 0.0154797
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020610, 0.0020485
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115686, 0.0116392
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032141, 0.0032337
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029174, 0.0029352
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108873, 0.0109538
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085254, 0.0084736
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007311, 0.0007355

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019309, upper bound: 0.0019476
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019074, upper bound: 0.0019581
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075078, 0.0074250
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021167, 0.0020934
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156177, 0.0154455
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020668, 0.0020440
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115430, 0.0116717
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032070, 0.0032427
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029110, 0.0029434
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108633, 0.0109844
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085491, 0.0084549
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007294, 0.0007376

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0019611
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019257, upper bound: 0.0019661
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0075172, 0.0074163
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0021194, 0.0020909
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0156373, 0.0154274
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0020693, 0.0020416
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115295, 0.0116863
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032032, 0.0032468
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029076, 0.0029471
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108505, 0.0109981
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0085599, 0.0084450
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007286, 0.0007385

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019276, upper bound: 0.0019821
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0019273, upper bound: 0.0019830
time: 1.52 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.17 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019817, upper bound: 0.0018679
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019157, upper bound: 0.0019223
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0018610, upper bound: 0.0018089
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0018608, upper bound: 0.0018089
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0018507, upper bound: 0.0018163
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0018507, upper bound: 0.0018163
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019694, upper bound: 0.0019197
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019511, upper bound: 0.0019381
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019692, upper bound: 0.0019352
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019680, upper bound: 0.0019396
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0020335, upper bound: 0.0019338
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0020190, upper bound: 0.0019423
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019900, upper bound: 0.0019267
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019823, upper bound: 0.0019306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019759, upper bound: 0.0019401
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019759, upper bound: 0.0019419
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019650, upper bound: 0.0019507
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019642, upper bound: 0.0019512
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019864, upper bound: 0.0019158
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0019175
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019105, upper bound: 0.0019457
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019007, upper bound: 0.0019555
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0020628, upper bound: 0.0019246
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0020549, upper bound: 0.0019260
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0020599, upper bound: 0.0019099
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0020423, upper bound: 0.0019392
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0014733, upper bound: 0.0014528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0014733, upper bound: 0.0014528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0014732, upper bound: 0.0014528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0014732, upper bound: 0.0014528
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019442, upper bound: 0.0018866
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019442, upper bound: 0.0018866
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0017343, upper bound: 0.0017134
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0017343, upper bound: 0.0017134
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019309, upper bound: 0.0019476
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019074, upper bound: 0.0019581
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019305, upper bound: 0.0019611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019257, upper bound: 0.0019661
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019276, upper bound: 0.0019821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 5, lower bound: -0.0019273, upper bound: 0.0019830

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0072212, 0.0074380
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020359, 0.0020970
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0150216, 0.0154725
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019879, 0.0020475
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0115632, 0.0112262
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0032126, 0.0031190
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0029161, 0.0028311
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0108822, 0.0105651
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0082229, 0.0084697
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007307, 0.0007094

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017846, upper bound: 0.0017028
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0017846, upper bound: 0.0017028
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071848, 0.0073175
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020257, 0.0020631
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149458, 0.0152219
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019778, 0.0020144
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113759, 0.0111696
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031606, 0.0031032
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028688, 0.0028168
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0107060, 0.0105118
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081814, 0.0083325
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007189, 0.0007058

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014069, upper bound: 0.0013865
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014069, upper bound: 0.0013865
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071940, 0.0073077
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020283, 0.0020603
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149650, 0.0152014
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019804, 0.0020117
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0113606, 0.0111839
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0031563, 0.0031072
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028650, 0.0028204
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0106916, 0.0105253
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081919, 0.0083213
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007179, 0.0007068

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0018016
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0018700, upper bound: 0.0018038
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071531, 0.0071684
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020167, 0.0020211
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0148799, 0.0149118
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019691, 0.0019733
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111442, 0.0111203
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0030962, 0.0030895
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028104, 0.0028044
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0104879, 0.0104654
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081453, 0.0081628
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007042, 0.0007027

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019759, upper bound: 0.0019065
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019752, upper bound: 0.0019117
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0156925, -0.0060777, -0.0156925, -0.0060777, -0.0071740, 0.0071479
1: -0.0073630, -0.0046522, -0.0073630, -0.0046522, -0.0020226, 0.0020152
2: -0.0157658, 0.0042350, -0.0157658, 0.0042350, -0.0149234, 0.0148690
3: -0.0004591, 0.0021877, -0.0004591, 0.0021877, -0.0019749, 0.0019677
4: 0.0029269, 0.0178742, 0.0029269, 0.0178742, -0.0111122, 0.0111528
5: 0.9963195, 1.0004723, 0.9963195, 1.0004723, -0.0030873, 0.0030986
6: 0.0045428, 0.0083123, 0.0045428, 0.0083123, -0.0028023, 0.0028126
7: -0.0064286, 0.0076385, -0.0064286, 0.0076385, -0.0104578, 0.0104961
8: -0.0151379, -0.0041895, -0.0151379, -0.0041895, -0.0081691, 0.0081393
9: -0.0036483, -0.0027037, -0.0036483, -0.0027037, -0.0007022, 0.0007048

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019008, upper bound: 0.0018567
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0019008, upper bound: 0.0018567
time: 2.17 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 5.22 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0017846, upper bound: 0.0017028
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0017846, upper bound: 0.0017028
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0014069, upper bound: 0.0013865
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0014069, upper bound: 0.0013865
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0018719, upper bound: 0.0018016
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0018700, upper bound: 0.0018038
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0019759, upper bound: 0.0019065
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0019752, upper bound: 0.0019117
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0019008, upper bound: 0.0018567
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.22
Output dim: 5, lower bound: -0.0019008, upper bound: 0.0018567
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0019864, upper bound: 0.0019158
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0019860, upper bound: 0.0019175
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0020628, upper bound: 0.0019246
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0020549, upper bound: 0.0019260
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0020599, upper bound: 0.0019099
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0020423, upper bound: 0.0019392
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0019276, upper bound: 0.0019821
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 5.22
Output dim: 5, lower bound: -0.0019273, upper bound: 0.0019830

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.47 + 597.47 = 600.95 seconds
