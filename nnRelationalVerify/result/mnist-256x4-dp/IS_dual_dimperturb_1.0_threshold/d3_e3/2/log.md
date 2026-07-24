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
Threshold: 27.4911861951


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255)
1: (-20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821)
2: (-25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841)
3: (-29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502)
4: (-27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682)
5: (-20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750)
6: (-22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550)
7: (-26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649)
8: (-32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634)
9: (-19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 15.60 = 17.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187042

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5002289, upper bound: 27.4913159
time: 6.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -27.4906210, upper bound: 27.4906208
time: 11.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.43 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.43
Output dim: 1, lower bound: -27.5002289, upper bound: 27.4913159
IS_A2, status: Status.VERIFIED, split count: 1, time: 18.43
Output dim: 1, lower bound: -27.4906210, upper bound: 27.4906208

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -21.4837723, 19.3528214, -21.4847088, 19.3536148, -40.8373871, 40.8375320
1: -20.9068127, 13.3130417, -20.9076710, 13.3137093, -34.2205200, 34.2207108
2: -25.2006702, 16.5126133, -25.2017784, 16.5133114, -41.7139778, 41.7143860
3: -29.8766422, 14.4121170, -29.8778210, 14.4127302, -44.2893715, 44.2899399
4: -27.2297478, 17.5693550, -27.2308311, 17.5701427, -44.7998886, 44.8001823
5: -20.8875847, 18.7882462, -20.8884811, 18.7889977, -39.6765823, 39.6767273
6: -22.1683369, 20.1590481, -22.1692429, 20.1599159, -42.3282547, 42.3282776
7: -26.5584335, 19.4229622, -26.5594749, 19.4237938, -45.9822273, 45.9824371
8: -32.1188736, 16.2818527, -32.1201401, 16.2826233, -48.4014969, 48.4019852
9: -19.4106407, 21.8066158, -19.4115181, 21.8075218, -41.2181587, 41.2181244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=55, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990883, upper bound: 27.4910726
time: 6.77 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5001043, upper bound: 27.4911426
time: 6.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 35.12 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 35.12
Output dim: 1, lower bound: -27.4990883, upper bound: 27.4910726
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 35.12
Output dim: 1, lower bound: -27.5001043, upper bound: 27.4911426

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -19.6625652, 17.7660809, -21.0976067, 19.0139198, -38.6764793, 38.8636856
1: -19.2144890, 12.1390791, -20.5490456, 13.0592403, -32.2737274, 32.6881218
2: -23.0105591, 15.1404247, -24.7297344, 16.2194138, -39.2299690, 39.8701591
3: -27.4335861, 13.2171955, -29.3611565, 14.1569128, -41.5904999, 42.5783539
4: -25.0155983, 16.0774250, -26.7621040, 17.2495842, -42.2651825, 42.8395233
5: -19.1220512, 17.2703362, -20.5094337, 18.4684315, -37.5904846, 37.7797699
6: -20.3430214, 18.4455357, -21.7817841, 19.7934074, -40.1364288, 40.2273178
7: -24.4038162, 17.8017311, -26.1034050, 19.0757313, -43.4795456, 43.9051361
8: -29.5399456, 14.8529701, -31.5729637, 15.9751129, -45.5150604, 46.4259338
9: -17.7372379, 20.0074482, -19.0509644, 21.4250259, -39.1622620, 39.0584106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=121, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=44, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4971444, upper bound: 27.4900643
time: 10.93 seconds

## Relational analysis of IS_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4944787, upper bound: 27.4883378
time: 10.47 seconds

## Relational analysis of IS_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990505, upper bound: 27.4910680
time: 7.05 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4987895, upper bound: 27.4905756
time: 12.64 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -20.4349174, 18.4345703, -21.2573528, 19.1531410, -39.5880585, 39.6919212
1: -19.9352131, 12.6261082, -20.6963921, 13.1629295, -33.0981407, 33.3225021
2: -23.9188232, 15.7141304, -24.9225559, 16.3400841, -40.2589073, 40.6366882
3: -28.4779053, 13.7185202, -29.5746632, 14.2623472, -42.7402534, 43.2931824
4: -25.9558067, 16.7000751, -26.9546700, 17.3806915, -43.3364983, 43.6547394
5: -19.8635540, 17.9158688, -20.6651192, 18.6005878, -38.4641418, 38.5809822
6: -21.1164379, 19.1637421, -21.9409580, 19.9438438, -41.0602798, 41.1046906
7: -25.3120823, 18.4784775, -26.2897835, 19.2179947, -44.5300751, 44.7682533
8: -30.6300125, 15.4489555, -31.7974243, 16.1013832, -46.7313919, 47.2463684
9: -18.4361153, 20.7675247, -19.1988068, 21.5819092, -40.0180206, 39.9663315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=122, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=42, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4982518, upper bound: 27.4901416
time: 6.40 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4961573, upper bound: 27.4884605
time: 11.63 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4971928, upper bound: 27.4907080
time: 14.16 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5000227, upper bound: 27.4911384
time: 9.56 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5000181, upper bound: 27.4911382
time: 11.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 125.75 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 125.75
Output dim: 1, lower bound: -27.4990505, upper bound: 27.4910680
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 125.75
Output dim: 1, lower bound: -27.4987895, upper bound: 27.4905756
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 125.75
Output dim: 1, lower bound: -27.5000227, upper bound: 27.4911384
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 125.75
Output dim: 1, lower bound: -27.5000181, upper bound: 27.4911382

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -19.5014248, 17.6239414, -20.2970371, 18.3062553, -37.8076782, 37.9209747
1: -19.0633602, 12.0380621, -19.7991886, 12.5558128, -31.6191730, 31.8372498
2: -22.8152885, 15.0182257, -23.7575951, 15.6108627, -38.4261436, 38.7758141
3: -27.2123146, 13.1124668, -28.2633476, 13.6352749, -40.8475876, 41.3758163
4: -24.8161049, 15.9449730, -25.7682838, 16.5898914, -41.4059944, 41.7132530
5: -18.9660816, 17.1329479, -19.7311783, 17.7861900, -36.7522736, 36.8641243
6: -20.1789837, 18.2952328, -20.9646416, 19.0454063, -39.2243843, 39.2598724
7: -24.2051697, 17.6573906, -25.1162376, 18.3564377, -42.5616074, 42.7736282
8: -29.3060665, 14.7288465, -30.4093838, 15.3580341, -44.6641006, 45.1382294
9: -17.5903835, 19.8455582, -18.3171043, 20.6199951, -38.2103767, 38.1626625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=121, inp2_unstable=122, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=42, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4971444, upper bound: 27.4900642
time: 19.49 seconds

## Relational analysis of IS_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4987895, upper bound: 27.4905756
time: 6.78 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4987895, upper bound: 27.4905756
time: 10.43 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -19.4594498, 17.5873032, -21.7423515, 19.5761261, -39.0355721, 39.3296547
1: -19.0236092, 12.0107050, -21.1327839, 13.3998318, -32.4234314, 33.1434898
2: -22.7648220, 14.9861660, -25.4397850, 16.6685677, -39.4333878, 40.4259491
3: -27.1555996, 13.0852346, -30.2722816, 14.5601368, -41.7157364, 43.3575172
4: -24.7650833, 15.9101429, -27.5880775, 17.7462425, -42.5113258, 43.4982224
5: -18.9255257, 17.0972881, -21.1178207, 19.0223160, -37.9478416, 38.2151108
6: -20.1365376, 18.2557774, -22.4369755, 20.3719807, -40.5085144, 40.6927528
7: -24.1547432, 17.6193790, -26.8748436, 19.6158676, -43.7706070, 44.4942245
8: -29.2458954, 14.6958323, -32.5298462, 16.4161015, -45.6619911, 47.2256584
9: -17.5515423, 19.8038406, -19.6091137, 22.0543957, -39.6059380, 39.4129486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=54, inp2_unstable=55, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=121, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=44, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4969036, upper bound: 27.4895498
time: 12.64 seconds

## Relational analysis of IS_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4940964, upper bound: 27.4877846
time: 8.15 seconds

## Relational analysis of IS_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4987895, upper bound: 27.4905757
time: 9.01 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4987895, upper bound: 27.4905757
time: 9.02 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -18.3842850, 16.6324654, -20.9313145, 18.8657455, -37.2500229, 37.5637817
1: -18.0433884, 11.4155884, -20.3960896, 12.9671535, -31.0105419, 31.8116741
2: -21.5202980, 14.1983662, -24.5363579, 16.0991592, -37.6194572, 38.7347221
3: -25.6217308, 12.4021902, -29.1239357, 14.0529785, -39.6747055, 41.5261230
4: -23.3852482, 15.0573416, -26.5457420, 17.1182747, -40.5035210, 41.6030807
5: -17.8920250, 16.1621628, -20.3489323, 18.3242493, -36.2162743, 36.5110931
6: -19.0346298, 17.2779236, -21.6096649, 19.6435947, -38.6782227, 38.8875885
7: -22.8266621, 16.6818447, -25.8944454, 18.9305782, -41.7572365, 42.5762863
8: -27.6376114, 13.9356012, -31.3234539, 15.8588400, -43.4964485, 45.2590561
9: -16.5944481, 18.7336082, -18.9027233, 21.2583694, -37.8528175, 37.6363297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=122, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4981724, upper bound: 27.4901376
time: 11.55 seconds

## Relational analysis of IS_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4961539, upper bound: 27.4884593
time: 8.03 seconds

## Relational analysis of IS_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4997809, upper bound: 27.4911203
time: 12.53 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4995267, upper bound: 27.4906258
time: 12.17 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -19.7549973, 17.8401356, -21.1122093, 19.0252914, -38.7802887, 38.9523430
1: -19.3122215, 12.2164564, -20.5631218, 13.0746136, -32.3868332, 32.7795715
2: -23.1159630, 15.2133713, -24.7499275, 16.2330341, -39.3489952, 39.9632950
3: -27.5400829, 13.2836523, -29.3747730, 14.1690521, -41.7091370, 42.6584244
4: -25.1078606, 16.1538143, -26.7726383, 17.2633533, -42.3712120, 42.9264526
5: -19.2090168, 17.3415241, -20.5241432, 18.4780216, -37.6870384, 37.8656693
6: -20.4299164, 18.5387726, -21.7937641, 19.8100243, -40.2399406, 40.3325348
7: -24.4923038, 17.8806190, -26.1144905, 19.0896378, -43.5819397, 43.9951096
8: -29.6500072, 14.9392586, -31.5875797, 15.9924965, -45.6425018, 46.5268364
9: -17.8199463, 20.0952530, -19.0659294, 21.4379463, -39.2578926, 39.1611786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=53, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=121, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=45, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 17.12 + 587.34 = 604.46 seconds
