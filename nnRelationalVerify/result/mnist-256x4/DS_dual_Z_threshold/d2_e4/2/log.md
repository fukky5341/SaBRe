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
execution time: IAR + RelationalAnalysis = 1.93 + 8.67 = 10.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -14.5592858, upper bound: 14.5592858

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
time: 3.92 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.46
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.46
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561

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

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
time: 7.45 seconds

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

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
time: 6.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 12.47 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.47
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.47
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.47
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.47
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
time: 3.35 seconds

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

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
time: 3.68 seconds

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

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
time: 2.15 seconds

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956
time: 10.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956
time: 6.63 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 18.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
time: 3.02 seconds

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
time: 2.96 seconds

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
time: 3.85 seconds

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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
time: 5.25 seconds

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

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
time: 5.65 seconds

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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
time: 6.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
time: 4.34 seconds

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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280
time: 2.68 seconds

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280
time: 2.68 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 8.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 8.87
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 10.60 + 166.92 = 177.52 seconds
