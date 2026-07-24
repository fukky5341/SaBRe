## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00368946


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853)
1: (-0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231)
2: (-0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180)
3: (-0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702)
4: (-0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449)
5: (-0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785)
6: (0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639)
7: (-0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929)
8: (-0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878)
9: (-0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.95 + 3.68 = 5.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0040994, upper bound: 0.0040994

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040573, upper bound: 0.0040520
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0040520, upper bound: 0.0040573
time: 2.90 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.09
Output dim: 6, lower bound: -0.0040573, upper bound: 0.0040520
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.09
Output dim: 6, lower bound: -0.0040520, upper bound: 0.0040573

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037423, upper bound: 0.0037344
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037378, upper bound: 0.0037401
time: 2.75 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037401, upper bound: 0.0037378
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037344, upper bound: 0.0037423
time: 2.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.16 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 6, lower bound: -0.0037423, upper bound: 0.0037344
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 6, lower bound: -0.0037378, upper bound: 0.0037401
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 6, lower bound: -0.0037401, upper bound: 0.0037378
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.16
Output dim: 6, lower bound: -0.0037344, upper bound: 0.0037423

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037216, upper bound: 0.0036957
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037013, upper bound: 0.0037134
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037174, upper bound: 0.0037004
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036969, upper bound: 0.0037189
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037189, upper bound: 0.0036969
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037004, upper bound: 0.0037174
time: 2.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037134, upper bound: 0.0037013
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036957, upper bound: 0.0037216
time: 2.64 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0037216, upper bound: 0.0036957
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0037013, upper bound: 0.0037134
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0037174, upper bound: 0.0037004
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0036969, upper bound: 0.0037189
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0037189, upper bound: 0.0036969
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0037004, upper bound: 0.0037174
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0037134, upper bound: 0.0037013
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.85
Output dim: 6, lower bound: -0.0036957, upper bound: 0.0037216

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036780, upper bound: 0.0035897
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036036, upper bound: 0.0036482
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036547, upper bound: 0.0035993
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035932, upper bound: 0.0036688
time: 2.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036739, upper bound: 0.0035931
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035996, upper bound: 0.0036533
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036494, upper bound: 0.0036035
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035900, upper bound: 0.0036741
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036741, upper bound: 0.0035901
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036035, upper bound: 0.0036494
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036533, upper bound: 0.0035997
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035931, upper bound: 0.0036739
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036688, upper bound: 0.0035932
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035993, upper bound: 0.0036547
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0057215, 0.0077068, 0.0057215, 0.0077068, -0.0019853, 0.0019853
1: -0.0017136, 0.0030095, -0.0017136, 0.0030095, -0.0047231, 0.0047231
2: -0.0073460, 0.0247720, -0.0073460, 0.0247720, -0.0321180, 0.0321180
3: -0.0046267, -0.0018565, -0.0046267, -0.0018565, -0.0027702, 0.0027702
4: -0.0012160, 0.0123290, -0.0012160, 0.0123290, -0.0135449, 0.0135449
5: -0.0023602, 0.0007184, -0.0023602, 0.0007184, -0.0030785, 0.0030785
6: 0.9888180, 0.9945819, 0.9888180, 0.9945819, -0.0057639, 0.0057639
7: -0.0157582, 0.0089347, -0.0157582, 0.0089347, -0.0246929, 0.0246929
8: -0.0092003, 0.0037875, -0.0092003, 0.0037875, -0.0129878, 0.0129878
9: -0.0148885, 0.0010365, -0.0148885, 0.0010365, -0.0159250, 0.0159250

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036482, upper bound: 0.0036036
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035897, upper bound: 0.0036779
time: 2.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 7.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036780, upper bound: 0.0035897
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036036, upper bound: 0.0036482
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036547, upper bound: 0.0035993
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0035932, upper bound: 0.0036688
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036739, upper bound: 0.0035931
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0035996, upper bound: 0.0036533
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036494, upper bound: 0.0036035
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0035900, upper bound: 0.0036741
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036741, upper bound: 0.0035901
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036035, upper bound: 0.0036494
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036533, upper bound: 0.0035997
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0035931, upper bound: 0.0036739
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036688, upper bound: 0.0035932
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0035993, upper bound: 0.0036547
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0036482, upper bound: 0.0036036
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 7.08
Output dim: 6, lower bound: -0.0035897, upper bound: 0.0036779

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.63 + 103.32 = 108.95 seconds
