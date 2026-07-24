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
execution time: IAR + RelationalAnalysis = 0.83 + 3.64 = 4.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0040994, upper bound: 0.0040994

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039844, upper bound: 0.0039626
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039626, upper bound: 0.0039844
time: 2.62 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.05 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.05
Output dim: 6, lower bound: -0.0039844, upper bound: 0.0039626
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.05
Output dim: 6, lower bound: -0.0039626, upper bound: 0.0039844

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039797, upper bound: 0.0039523
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039771, upper bound: 0.0039579
time: 2.33 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039492, upper bound: 0.0039243
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039075, upper bound: 0.0039713
time: 2.42 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.37 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.37
Output dim: 6, lower bound: -0.0039797, upper bound: 0.0039523
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.37
Output dim: 6, lower bound: -0.0039771, upper bound: 0.0039579
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.37
Output dim: 6, lower bound: -0.0039492, upper bound: 0.0039243
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.37
Output dim: 6, lower bound: -0.0039075, upper bound: 0.0039713

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039230, upper bound: 0.0038621
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038814, upper bound: 0.0038953
time: 2.37 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039484, upper bound: 0.0039200
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039260, upper bound: 0.0039294
time: 2.33 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039445, upper bound: 0.0039195
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039384, upper bound: 0.0039197
time: 2.96 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038791, upper bound: 0.0039222
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038644, upper bound: 0.0039427
time: 2.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.41 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0039230, upper bound: 0.0038621
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0038814, upper bound: 0.0038953
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0039484, upper bound: 0.0039200
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0039260, upper bound: 0.0039294
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0039445, upper bound: 0.0039195
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0039384, upper bound: 0.0039197
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0038791, upper bound: 0.0039222
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.41
Output dim: 6, lower bound: -0.0038644, upper bound: 0.0039427

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039229, upper bound: 0.0038556
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0039187, upper bound: 0.0038621
time: 3.12 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038579, upper bound: 0.0038577
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038360, upper bound: 0.0038689
time: 2.56 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036117, upper bound: 0.0036118
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036117, upper bound: 0.0036118
time: 2.35 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033751, upper bound: 0.0033785
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033751, upper bound: 0.0033785
time: 1.34 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038989, upper bound: 0.0038102
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038389, upper bound: 0.0038761
time: 2.53 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038954, upper bound: 0.0038789
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038940, upper bound: 0.0038790
time: 2.93 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038380, upper bound: 0.0038188
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037958, upper bound: 0.0038859
time: 2.59 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038511, upper bound: 0.0039013
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038364, upper bound: 0.0039301
time: 2.44 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0039229, upper bound: 0.0038556
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0039187, upper bound: 0.0038621
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038579, upper bound: 0.0038577
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038360, upper bound: 0.0038689
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0036117, upper bound: 0.0036118
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0036117, upper bound: 0.0036118
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0033751, upper bound: 0.0033785
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0033751, upper bound: 0.0033785
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038989, upper bound: 0.0038102
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038389, upper bound: 0.0038761
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038954, upper bound: 0.0038789
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038940, upper bound: 0.0038790
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038380, upper bound: 0.0038188
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0037958, upper bound: 0.0038859
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038511, upper bound: 0.0039013
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 6, lower bound: -0.0038364, upper bound: 0.0039301

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038811, upper bound: 0.0038067
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038811, upper bound: 0.0038067
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038780, upper bound: 0.0038117
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038780, upper bound: 0.0038117
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037547, upper bound: 0.0037568
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037527, upper bound: 0.0037616
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038250, upper bound: 0.0038579
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038250, upper bound: 0.0038579
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036327, upper bound: 0.0035389
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036289, upper bound: 0.0035433
time: 2.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038113, upper bound: 0.0038196
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038086, upper bound: 0.0038479
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038057, upper bound: 0.0037880
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038005, upper bound: 0.0037896
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033923, upper bound: 0.0033661
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033923, upper bound: 0.0033661
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037958, upper bound: 0.0037752
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037958, upper bound: 0.0037753
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037839, upper bound: 0.0038749
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037833, upper bound: 0.0038747
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038402, upper bound: 0.0038906
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038257, upper bound: 0.0038906
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038145, upper bound: 0.0038868
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038062, upper bound: 0.0039104
time: 2.36 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038811, upper bound: 0.0038067
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038811, upper bound: 0.0038067
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038780, upper bound: 0.0038117
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038780, upper bound: 0.0038117
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0037547, upper bound: 0.0037568
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0037527, upper bound: 0.0037616
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038250, upper bound: 0.0038579
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038250, upper bound: 0.0038579
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0036327, upper bound: 0.0035389
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0036289, upper bound: 0.0035433
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038113, upper bound: 0.0038196
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038086, upper bound: 0.0038479
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038057, upper bound: 0.0037880
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038005, upper bound: 0.0037896
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0033923, upper bound: 0.0033661
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0033923, upper bound: 0.0033661
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0037958, upper bound: 0.0037752
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0037958, upper bound: 0.0037753
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0037839, upper bound: 0.0038749
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0037833, upper bound: 0.0038747
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038402, upper bound: 0.0038906
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038257, upper bound: 0.0038906
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038145, upper bound: 0.0038868
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.43
Output dim: 6, lower bound: -0.0038062, upper bound: 0.0039104

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037055, upper bound: 0.0036444
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037055, upper bound: 0.0036444
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035946, upper bound: 0.0035532
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035946, upper bound: 0.0035532
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038416, upper bound: 0.0037250
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037679, upper bound: 0.0037705
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038416, upper bound: 0.0037250
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037679, upper bound: 0.0037705
time: 2.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037137, upper bound: 0.0037176
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037118, upper bound: 0.0037200
time: 2.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037361, upper bound: 0.0037111
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037191, upper bound: 0.0037472
time: 2.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037253, upper bound: 0.0037583
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037237, upper bound: 0.0037610
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036817, upper bound: 0.0037066
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036817, upper bound: 0.0037066
time: 2.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038114, upper bound: 0.0038177
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038074, upper bound: 0.0038196
time: 2.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037682, upper bound: 0.0038043
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037683, upper bound: 0.0038043
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037703, upper bound: 0.0037496
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037703, upper bound: 0.0037497
time: 2.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037612, upper bound: 0.0036938
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037030, upper bound: 0.0037501
time: 2.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037586, upper bound: 0.0037357
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037578, upper bound: 0.0037371
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035029, upper bound: 0.0034818
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035003, upper bound: 0.0034829
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037392, upper bound: 0.0037734
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036804, upper bound: 0.0038287
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037100, upper bound: 0.0038127
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037100, upper bound: 0.0038127
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038354, upper bound: 0.0038854
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038334, upper bound: 0.0038859
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037999, upper bound: 0.0037934
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037540, upper bound: 0.0038531
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037733, upper bound: 0.0038410
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037733, upper bound: 0.0038437
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038062, upper bound: 0.0039081
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038059, upper bound: 0.0039104
time: 1.95 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037055, upper bound: 0.0036444
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037055, upper bound: 0.0036444
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0035946, upper bound: 0.0035532
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0035946, upper bound: 0.0035532
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038416, upper bound: 0.0037250
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037679, upper bound: 0.0037705
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038416, upper bound: 0.0037250
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037679, upper bound: 0.0037705
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037137, upper bound: 0.0037176
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037118, upper bound: 0.0037200
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037361, upper bound: 0.0037111
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037191, upper bound: 0.0037472
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037253, upper bound: 0.0037583
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037237, upper bound: 0.0037610
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0036817, upper bound: 0.0037066
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0036817, upper bound: 0.0037066
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038114, upper bound: 0.0038177
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038074, upper bound: 0.0038196
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037682, upper bound: 0.0038043
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037683, upper bound: 0.0038043
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037703, upper bound: 0.0037496
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037703, upper bound: 0.0037497
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037612, upper bound: 0.0036938
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037030, upper bound: 0.0037501
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037586, upper bound: 0.0037357
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037578, upper bound: 0.0037371
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0035029, upper bound: 0.0034818
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0035003, upper bound: 0.0034829
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037392, upper bound: 0.0037734
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0036804, upper bound: 0.0038287
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037100, upper bound: 0.0038127
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037100, upper bound: 0.0038127
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038354, upper bound: 0.0038854
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038334, upper bound: 0.0038859
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037999, upper bound: 0.0037934
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037540, upper bound: 0.0038531
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037733, upper bound: 0.0038410
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0037733, upper bound: 0.0038437
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038062, upper bound: 0.0039081
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.16
Output dim: 6, lower bound: -0.0038059, upper bound: 0.0039104

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035953, upper bound: 0.0035601
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035953, upper bound: 0.0036025
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036665, upper bound: 0.0036004
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036518, upper bound: 0.0036067
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037401, upper bound: 0.0036278
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037381, upper bound: 0.0036281
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037358, upper bound: 0.0037052
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036896, upper bound: 0.0037336
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0036977
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037841, upper bound: 0.0037085
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037045
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037045
time: 2.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0036261
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036007, upper bound: 0.0036658
time: 3.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035890, upper bound: 0.0035985
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035890, upper bound: 0.0035985
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037150, upper bound: 0.0036735
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0036902
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036802, upper bound: 0.0037059
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036795, upper bound: 0.0037088
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036852, upper bound: 0.0037163
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036839, upper bound: 0.0037199
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036774, upper bound: 0.0036606
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036273, upper bound: 0.0037142
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036659, upper bound: 0.0036589
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036414, upper bound: 0.0036924
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036599, upper bound: 0.0036712
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036485, upper bound: 0.0036848
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035562, upper bound: 0.0035618
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035561, upper bound: 0.0035622
time: 2.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037427, upper bound: 0.0037571
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037408, upper bound: 0.0037585
time: 2.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037302, upper bound: 0.0037206
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037145, upper bound: 0.0037742
time: 2.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037302, upper bound: 0.0037206
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036779, upper bound: 0.0037743
time: 2.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037373, upper bound: 0.0036729
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0037171
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036484, upper bound: 0.0036281
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036484, upper bound: 0.0036281
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036310, upper bound: 0.0035607
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036310, upper bound: 0.0035607
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036827, upper bound: 0.0037099
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036659, upper bound: 0.0037296
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037586, upper bound: 0.0037349
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037567, upper bound: 0.0037357
time: 2.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032205, upper bound: 0.0032232
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0032205, upper bound: 0.0032232
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036261, upper bound: 0.0036560
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036248, upper bound: 0.0036579
time: 2.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033768, upper bound: 0.0034946
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0033768, upper bound: 0.0034946
time: 2.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035980, upper bound: 0.0036872
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0035980, upper bound: 0.0036908
time: 4.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036506, upper bound: 0.0037534
time: 2.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036506, upper bound: 0.0037534
time: 2.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037848, upper bound: 0.0037801
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037484, upper bound: 0.0038416
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037924, upper bound: 0.0037875
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037497, upper bound: 0.0038492
time: 2.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037342, upper bound: 0.0037212
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037342, upper bound: 0.0037211
time: 2.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036367, upper bound: 0.0037362
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036358, upper bound: 0.0037371
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037197, upper bound: 0.0037825
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037188, upper bound: 0.0037832
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036710, upper bound: 0.0037285
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036708, upper bound: 0.0037314
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037521, upper bound: 0.0038062
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037369, upper bound: 0.0038538
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037713, upper bound: 0.0038327
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037363, upper bound: 0.0038789
time: 2.19 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035953, upper bound: 0.0035601
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035953, upper bound: 0.0036025
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036665, upper bound: 0.0036004
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036518, upper bound: 0.0036067
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037401, upper bound: 0.0036278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037381, upper bound: 0.0036281
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037358, upper bound: 0.0037052
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036896, upper bound: 0.0037336
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0038284, upper bound: 0.0036977
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037841, upper bound: 0.0037085
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037045
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0037045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036721, upper bound: 0.0036261
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036007, upper bound: 0.0036658
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035890, upper bound: 0.0035985
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035890, upper bound: 0.0035985
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037150, upper bound: 0.0036735
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037012, upper bound: 0.0036902
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036802, upper bound: 0.0037059
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036795, upper bound: 0.0037088
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036852, upper bound: 0.0037163
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036839, upper bound: 0.0037199
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036774, upper bound: 0.0036606
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036273, upper bound: 0.0037142
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036659, upper bound: 0.0036589
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036414, upper bound: 0.0036924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036599, upper bound: 0.0036712
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036485, upper bound: 0.0036848
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035562, upper bound: 0.0035618
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035561, upper bound: 0.0035622
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037427, upper bound: 0.0037571
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037408, upper bound: 0.0037585
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037302, upper bound: 0.0037206
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037145, upper bound: 0.0037742
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037302, upper bound: 0.0037206
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036779, upper bound: 0.0037743
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037373, upper bound: 0.0036729
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0037171
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036484, upper bound: 0.0036281
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036484, upper bound: 0.0036281
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036310, upper bound: 0.0035607
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036310, upper bound: 0.0035607
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036827, upper bound: 0.0037099
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036659, upper bound: 0.0037296
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037586, upper bound: 0.0037349
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037567, upper bound: 0.0037357
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0032205, upper bound: 0.0032232
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0032205, upper bound: 0.0032232
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036261, upper bound: 0.0036560
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036248, upper bound: 0.0036579
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0033768, upper bound: 0.0034946
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0033768, upper bound: 0.0034946
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035980, upper bound: 0.0036872
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0035980, upper bound: 0.0036908
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036506, upper bound: 0.0037534
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036506, upper bound: 0.0037534
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037848, upper bound: 0.0037801
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037484, upper bound: 0.0038416
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037924, upper bound: 0.0037875
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037497, upper bound: 0.0038492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037342, upper bound: 0.0037212
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037342, upper bound: 0.0037211
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036367, upper bound: 0.0037362
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036358, upper bound: 0.0037371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037197, upper bound: 0.0037825
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037188, upper bound: 0.0037832
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036710, upper bound: 0.0037285
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0036708, upper bound: 0.0037314
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037521, upper bound: 0.0038062
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037369, upper bound: 0.0038538
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037713, upper bound: 0.0038327
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.56
Output dim: 6, lower bound: -0.0037363, upper bound: 0.0038789

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036759, upper bound: 0.0035708
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036059, upper bound: 0.0035708
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036920, upper bound: 0.0035915
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036274, upper bound: 0.0035921
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036785, upper bound: 0.0036788
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036910, upper bound: 0.0036809
time: 2.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036679, upper bound: 0.0037038
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036555, upper bound: 0.0037110
time: 2.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038082, upper bound: 0.0036676
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037842, upper bound: 0.0036745
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034964, upper bound: 0.0034592
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0034964, upper bound: 0.0034592
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036513, upper bound: 0.0036531
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036512, upper bound: 0.0036541
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036511, upper bound: 0.0036537
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036493, upper bound: 0.0036545
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037044, upper bound: 0.0036642
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037038, upper bound: 0.0036639
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036537, upper bound: 0.0036432
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036537, upper bound: 0.0036431
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036802, upper bound: 0.0037018
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036771, upper bound: 0.0037058
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036447, upper bound: 0.0036760
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036433, upper bound: 0.0036764
time: 2.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036696, upper bound: 0.0036754
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036518, upper bound: 0.0037012
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036518, upper bound: 0.0036499
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036115, upper bound: 0.0036867
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035907, upper bound: 0.0036328
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035679, upper bound: 0.0036828
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036000, upper bound: 0.0036471
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0035997, upper bound: 0.0036499
time: 2.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037080, upper bound: 0.0037221
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037082, upper bound: 0.0037220
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037021, upper bound: 0.0036768
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0036848, upper bound: 0.0037282
time: 1.76 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036759, upper bound: 0.0035708
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036059, upper bound: 0.0035708
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036920, upper bound: 0.0035915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036274, upper bound: 0.0035921
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036785, upper bound: 0.0036788
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036910, upper bound: 0.0036809
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036679, upper bound: 0.0037038
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036555, upper bound: 0.0037110
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0038082, upper bound: 0.0036676
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0037842, upper bound: 0.0036745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0034964, upper bound: 0.0034592
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0034964, upper bound: 0.0034592
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036513, upper bound: 0.0036531
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036512, upper bound: 0.0036541
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036511, upper bound: 0.0036537
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036493, upper bound: 0.0036545
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0037044, upper bound: 0.0036642
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0037038, upper bound: 0.0036639
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036537, upper bound: 0.0036432
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036537, upper bound: 0.0036431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036802, upper bound: 0.0037018
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036771, upper bound: 0.0037058
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036447, upper bound: 0.0036760
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036433, upper bound: 0.0036764
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036696, upper bound: 0.0036754
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036518, upper bound: 0.0037012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036518, upper bound: 0.0036499
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036115, upper bound: 0.0036867
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0035907, upper bound: 0.0036328
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0035679, upper bound: 0.0036828
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036000, upper bound: 0.0036471
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0035997, upper bound: 0.0036499
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0037080, upper bound: 0.0037221
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0037082, upper bound: 0.0037220
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0037021, upper bound: 0.0036768
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 6, lower bound: -0.0036848, upper bound: 0.0037282
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037302, upper bound: 0.0037206
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037145, upper bound: 0.0037742
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037302, upper bound: 0.0037206
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036779, upper bound: 0.0037743
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037373, upper bound: 0.0036729
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036994, upper bound: 0.0037171
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036827, upper bound: 0.0037099
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036659, upper bound: 0.0037296
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037586, upper bound: 0.0037349
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037567, upper bound: 0.0037357
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0035980, upper bound: 0.0036908
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036506, upper bound: 0.0037534
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036506, upper bound: 0.0037534
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037848, upper bound: 0.0037801
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037484, upper bound: 0.0038416
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037924, upper bound: 0.0037875
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037497, upper bound: 0.0038492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037342, upper bound: 0.0037212
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037342, upper bound: 0.0037211
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036367, upper bound: 0.0037362
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036358, upper bound: 0.0037371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037197, upper bound: 0.0037825
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037188, upper bound: 0.0037832
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036710, upper bound: 0.0037285
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0036708, upper bound: 0.0037314
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037521, upper bound: 0.0038062
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037369, upper bound: 0.0038538
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037713, upper bound: 0.0038327
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.40
Output dim: 6, lower bound: -0.0037363, upper bound: 0.0038789

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.48 + 597.49 = 601.97 seconds
