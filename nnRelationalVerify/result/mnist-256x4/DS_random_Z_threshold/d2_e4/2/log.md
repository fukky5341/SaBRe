## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 14.544726514199999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500)
1: (-8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229)
2: (-10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675)
3: (-11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024)
4: (-10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131)
5: (-9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238)
6: (-8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851)
7: (-10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482)
8: (-11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781)
9: (-8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.83 + 8.52 = 9.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -14.5592858, upper bound: 14.5592858

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574946, upper bound: 14.5574946
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574946, upper bound: 14.5574946
time: 2.83 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.70 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.70
Output dim: 7, lower bound: -14.5574946, upper bound: 14.5574946
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.70
Output dim: 7, lower bound: -14.5574946, upper bound: 14.5574946

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574480, upper bound: 14.5574478
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574478, upper bound: 14.5574480
time: 3.22 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574935, upper bound: 14.5574946
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574946, upper bound: 14.5574935
time: 3.50 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.14 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.14
Output dim: 7, lower bound: -14.5574480, upper bound: 14.5574478
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.14
Output dim: 7, lower bound: -14.5574478, upper bound: 14.5574480
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.14
Output dim: 7, lower bound: -14.5574935, upper bound: 14.5574946
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.14
Output dim: 7, lower bound: -14.5574946, upper bound: 14.5574935

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555604, upper bound: 14.5555603
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555604, upper bound: 14.5555603
time: 3.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574478, upper bound: 14.5574480
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5574478, upper bound: 14.5574480
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566460, upper bound: 14.5566471
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566460, upper bound: 14.5566471
time: 4.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5568907, upper bound: 14.5568892
time: 5.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5568920, upper bound: 14.5568878
time: 14.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 19.99 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5555604, upper bound: 14.5555603
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5555604, upper bound: 14.5555603
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5574478, upper bound: 14.5574480
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5574478, upper bound: 14.5574480
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5566460, upper bound: 14.5566471
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5566460, upper bound: 14.5566471
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5568907, upper bound: 14.5568892
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.99
Output dim: 7, lower bound: -14.5568920, upper bound: 14.5568878

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547861, upper bound: 14.5547866
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547862, upper bound: 14.5547865
time: 5.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555590, upper bound: 14.5555603
time: 7.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555604, upper bound: 14.5555590
time: 8.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485990
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485990
time: 4.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5572360, upper bound: 14.5572363
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5572360, upper bound: 14.5572363
time: 5.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5563881, upper bound: 14.5563923
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5563911, upper bound: 14.5563890
time: 5.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5443004, upper bound: 14.5443015
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5443004, upper bound: 14.5443015
time: 4.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5561134, upper bound: 14.5561132
time: 18.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5561158, upper bound: 14.5561099
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5561143, upper bound: 14.5561125
time: 4.51 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 10.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5547861, upper bound: 14.5547866
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5547862, upper bound: 14.5547865
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5555590, upper bound: 14.5555603
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5555604, upper bound: 14.5555590
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485990
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485990
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5572360, upper bound: 14.5572363
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5572360, upper bound: 14.5572363
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5563881, upper bound: 14.5563923
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5563911, upper bound: 14.5563890
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5443004, upper bound: 14.5443015
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5443004, upper bound: 14.5443015
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5561134, upper bound: 14.5561132
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5561158, upper bound: 14.5561099
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.74
Output dim: 7, lower bound: -14.5561143, upper bound: 14.5561125

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547861, upper bound: 14.5547864
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547860, upper bound: 14.5547866
time: 25.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546608, upper bound: 14.5546614
time: 6.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546612, upper bound: 14.5546610
time: 3.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5491113, upper bound: 14.5491128
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5491113, upper bound: 14.5491128
time: 6.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5548075, upper bound: 14.5548018
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5548060, upper bound: 14.5548043
time: 4.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5480050, upper bound: 14.5480033
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5480050, upper bound: 14.5480033
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485989
time: 2.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485990
time: 5.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5432242, upper bound: 14.5432255
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5432242, upper bound: 14.5432255
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5542938, upper bound: 14.5542981
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5542938, upper bound: 14.5542981
time: 3.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546207, upper bound: 14.5546236
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546207, upper bound: 14.5546236
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5554698, upper bound: 14.5554700
time: 6.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5554732, upper bound: 14.5554700
time: 6.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5553256, upper bound: 14.5553363
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5553256, upper bound: 14.5553363
time: 10.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547629, upper bound: 14.5547600
time: 4.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547629, upper bound: 14.5547600
time: 9.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5560697, upper bound: 14.5560659
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5560697, upper bound: 14.5560659
time: 3.42 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 8.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5547861, upper bound: 14.5547864
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5547860, upper bound: 14.5547866
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5546608, upper bound: 14.5546614
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5546612, upper bound: 14.5546610
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5491113, upper bound: 14.5491128
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5491113, upper bound: 14.5491128
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5548075, upper bound: 14.5548018
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5548060, upper bound: 14.5548043
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5480050, upper bound: 14.5480033
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5480050, upper bound: 14.5480033
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485989
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5486003, upper bound: 14.5485990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5432242, upper bound: 14.5432255
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5432242, upper bound: 14.5432255
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5542938, upper bound: 14.5542981
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5542938, upper bound: 14.5542981
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5546207, upper bound: 14.5546236
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5546207, upper bound: 14.5546236
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5554698, upper bound: 14.5554700
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5554732, upper bound: 14.5554700
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5553256, upper bound: 14.5553363
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5553256, upper bound: 14.5553363
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5547629, upper bound: 14.5547600
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5547629, upper bound: 14.5547600
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5560697, upper bound: 14.5560659
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.01
Output dim: 7, lower bound: -14.5560697, upper bound: 14.5560659

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5539975, upper bound: 14.5539993
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5539975, upper bound: 14.5539993
time: 3.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5531982, upper bound: 14.5532049
time: 6.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5531982, upper bound: 14.5532049
time: 6.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5369578, upper bound: 14.5369575
time: 8.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5369578, upper bound: 14.5369575
time: 8.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546018, upper bound: 14.5546020
time: 33.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546018, upper bound: 14.5546020
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5491113, upper bound: 14.5491120
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5491109, upper bound: 14.5491128
time: 3.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5481106, upper bound: 14.5481121
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5481101, upper bound: 14.5481120
time: 16.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5535741, upper bound: 14.5535710
time: 10.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5535739, upper bound: 14.5535711
time: 9.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5548060, upper bound: 14.5548026
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5548048, upper bound: 14.5548043
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5480049, upper bound: 14.5480033
time: 9.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5480050, upper bound: 14.5480033
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5469414, upper bound: 14.5469382
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5469414, upper bound: 14.5469382
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5485287, upper bound: 14.5485315
time: 6.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5485327, upper bound: 14.5485270
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5475393, upper bound: 14.5475381
time: 9.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5475386, upper bound: 14.5475381
time: 6.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5538398, upper bound: 14.5538410
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5538398, upper bound: 14.5538408
time: 12.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5534664, upper bound: 14.5534645
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5534635, upper bound: 14.5534685
time: 12.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546207, upper bound: 14.5546235
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546205, upper bound: 14.5546236
time: 3.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5476066, upper bound: 14.5476126
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5476066, upper bound: 14.5476126
time: 17.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5553176, upper bound: 14.5553169
time: 8.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5553177, upper bound: 14.5553170
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5544769, upper bound: 14.5544756
time: 7.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5544756, upper bound: 14.5544762
time: 10.10 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 18.75 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5539975, upper bound: 14.5539993
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5539975, upper bound: 14.5539993
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5531982, upper bound: 14.5532049
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5531982, upper bound: 14.5532049
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5369578, upper bound: 14.5369575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5369578, upper bound: 14.5369575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5546018, upper bound: 14.5546020
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5546018, upper bound: 14.5546020
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5491113, upper bound: 14.5491120
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5491109, upper bound: 14.5491128
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5481106, upper bound: 14.5481121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5481101, upper bound: 14.5481120
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5535741, upper bound: 14.5535710
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5535739, upper bound: 14.5535711
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5548060, upper bound: 14.5548026
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5548048, upper bound: 14.5548043
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5480049, upper bound: 14.5480033
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5480050, upper bound: 14.5480033
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5469414, upper bound: 14.5469382
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5469414, upper bound: 14.5469382
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5485287, upper bound: 14.5485315
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5485327, upper bound: 14.5485270
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5475393, upper bound: 14.5475381
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5475386, upper bound: 14.5475381
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5538398, upper bound: 14.5538410
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5538398, upper bound: 14.5538408
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5534664, upper bound: 14.5534645
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5534635, upper bound: 14.5534685
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5546207, upper bound: 14.5546235
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5546205, upper bound: 14.5546236
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5476066, upper bound: 14.5476126
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5476066, upper bound: 14.5476126
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5553176, upper bound: 14.5553169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5553177, upper bound: 14.5553170
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5544769, upper bound: 14.5544756
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.75
Output dim: 7, lower bound: -14.5544756, upper bound: 14.5544762
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5561150, upper bound: 14.5561106
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5553256, upper bound: 14.5553363
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5553256, upper bound: 14.5553363
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5547629, upper bound: 14.5547600
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5547629, upper bound: 14.5547600
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5560697, upper bound: 14.5560659
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 7, lower bound: -14.5560697, upper bound: 14.5560659

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 9.36 + 601.77 = 611.13 seconds
