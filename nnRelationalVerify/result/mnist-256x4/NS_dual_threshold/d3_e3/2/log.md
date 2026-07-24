## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 27.4911861951


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.77 + 15.41 = 16.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187042

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5002289, upper bound: 27.4913159
time: 6.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -27.4906210, upper bound: 27.4906208
time: 11.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 18.13 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 18.13
Output dim: 1, lower bound: -27.5002289, upper bound: 27.4913159
NS_A2, status: Status.VERIFIED, split count: 1, time: 18.13
Output dim: 1, lower bound: -27.4906210, upper bound: 27.4906208

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4983861, upper bound: 27.4902212
time: 8.02 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4961980, upper bound: 27.4885247
time: 7.37 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4972634, upper bound: 27.4908510
time: 24.14 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5000247, upper bound: 27.4913004
time: 9.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
time: 5.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 104.40 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 104.40
Output dim: 1, lower bound: -27.5000247, upper bound: 27.4913004
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 104.40
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -21.3159275, 19.2036953, -20.6746750, 18.6349678, -39.9508972, 39.8783684
1: -20.7494030, 13.2066927, -20.1485920, 12.8010101, -33.5504150, 33.3552856
2: -24.9961777, 16.3847504, -24.2154408, 15.8968525, -40.8930283, 40.6001892
3: -29.6462173, 14.3026447, -28.7672100, 13.8842430, -43.5304604, 43.0698509
4: -27.0209198, 17.4306259, -26.2246475, 16.9007721, -43.9216919, 43.6552696
5: -20.7237129, 18.6454430, -20.0977821, 18.1001015, -38.8238144, 38.7432251
6: -21.9966393, 20.0020504, -21.3413544, 19.4022255, -41.3988647, 41.3434067
7: -26.3514156, 19.2714691, -25.5614758, 18.6935883, -45.0450058, 44.8329468
8: -31.8743477, 16.1525383, -30.9418297, 15.6574249, -47.5317574, 47.0943680
9: -19.2558823, 21.6375809, -18.6652508, 20.9925251, -40.2484055, 40.3028297

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
time: 6.35 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
time: 7.63 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -21.2677383, 19.1611023, -22.1228905, 19.9092846, -41.1770172, 41.2839928
1: -20.7038612, 13.1750908, -21.4860020, 13.6509886, -34.3548470, 34.6610870
2: -24.9375572, 16.3478165, -25.9021721, 16.9558144, -41.8933678, 42.2499771
3: -29.5808239, 14.2712355, -30.7803001, 14.8109188, -44.3917427, 45.0515366
4: -26.9618034, 17.3904114, -28.0480099, 18.0603790, -45.0221786, 45.4384232
5: -20.6765671, 18.6045723, -21.4894619, 19.3367386, -40.0133018, 40.0940323
6: -21.9475346, 19.9565964, -22.8159809, 20.7313824, -42.6789169, 42.7725754
7: -26.2932892, 19.2274513, -27.3217888, 19.9563923, -46.2496796, 46.5492401
8: -31.8047695, 16.1147156, -33.0651932, 16.7184811, -48.5232506, 49.1799088
9: -19.2107582, 21.5894508, -19.9618492, 22.4300671, -41.6408081, 41.5513000

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
time: 8.24 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
time: 9.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 18.76 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.76
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.76
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.76
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.76
Output dim: 1, lower bound: -27.4997996, upper bound: 27.4907842

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -20.6737614, 18.6341972, -20.6746750, 18.6349678, -39.3087273, 39.3088722
1: -20.1477509, 12.8003759, -20.1485920, 12.8010101, -32.9487610, 32.9489670
2: -24.2143612, 15.8961754, -24.2154408, 15.8968525, -40.1112137, 40.1116142
3: -28.7660522, 13.8836441, -28.7672100, 13.8842430, -42.6502953, 42.6508522
4: -26.2235928, 16.9000053, -26.2246475, 16.9007721, -43.1243668, 43.1246529
5: -20.0969048, 18.0993633, -20.0977821, 18.1001015, -38.1970062, 38.1971436
6: -21.3404751, 19.4013748, -21.3413544, 19.4022255, -40.7426987, 40.7427216
7: -25.5604591, 18.6927795, -25.5614758, 18.6935883, -44.2540474, 44.2542496
8: -30.9406013, 15.6566696, -30.9418297, 15.6574249, -46.5980263, 46.5984993
9: -18.6644020, 20.9916458, -18.6652508, 20.9925251, -39.6569290, 39.6568985

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4998983, upper bound: 27.4910642
time: 6.98 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4992984, upper bound: 27.4910322
time: 16.12 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -22.1206589, 19.9069862, -20.6746750, 18.6349678, -40.7556267, 40.5816536
1: -21.4835072, 13.6481037, -20.1485920, 12.8010101, -34.2845154, 33.7966957
2: -25.8994961, 16.9538536, -24.2154408, 15.8968525, -41.7963486, 41.1692886
3: -30.7769737, 14.8093901, -28.7672100, 13.8842430, -44.6612129, 43.5765953
4: -28.0450630, 18.0584641, -26.2246475, 16.9007721, -44.9458351, 44.2831116
5: -21.4874229, 19.3349056, -20.0977821, 18.1001015, -39.5875244, 39.4326859
6: -22.8137627, 20.7293110, -21.3413544, 19.4022255, -42.2159882, 42.0706635
7: -27.3187599, 19.9542923, -25.5614758, 18.6935883, -46.0123482, 45.5157700
8: -33.0624084, 16.7163353, -30.9418297, 15.6574249, -48.7198296, 47.6581612
9: -19.9594021, 22.4274120, -18.6652508, 20.9925251, -40.9519272, 41.0926628

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4987598, upper bound: 27.4912247
time: 12.54 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5000247, upper bound: 27.4913001
time: 11.17 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -20.6737614, 18.6341972, -22.1228905, 19.9092846, -40.5830383, 40.7570877
1: -20.1477509, 12.8003759, -21.4860020, 13.6509886, -33.7987366, 34.2863731
2: -24.2143612, 15.8961754, -25.9021721, 16.9558144, -41.1701698, 41.7983360
3: -28.7660522, 13.8836441, -30.7803001, 14.8109188, -43.5769730, 44.6639442
4: -26.2235928, 16.9000053, -28.0480099, 18.0603790, -44.2839699, 44.9480133
5: -20.0969048, 18.0993633, -21.4894619, 19.3367386, -39.4336395, 39.5888252
6: -21.3404751, 19.4013748, -22.8159809, 20.7313824, -42.0718575, 42.2173538
7: -25.5604591, 18.6927795, -27.3217888, 19.9563923, -45.5168495, 46.0145645
8: -30.9406013, 15.6566696, -33.0651932, 16.7184811, -47.6590805, 48.7218590
9: -18.6644020, 20.9916458, -19.9618492, 22.4300671, -41.0944672, 40.9534950

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4994619, upper bound: 27.4902497
time: 15.92 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189
time: 9.69 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -22.1206589, 19.9069862, -22.1228905, 19.9092846, -42.0299377, 42.0298653
1: -21.4835072, 13.6481037, -21.4860020, 13.6509886, -35.1344910, 35.1341057
2: -25.8994961, 16.9538536, -25.9021721, 16.9558144, -42.8553085, 42.8560181
3: -30.7769737, 14.8093901, -30.7803001, 14.8109188, -45.5878906, 45.5896873
4: -28.0450630, 18.0584641, -28.0480099, 18.0603790, -46.1054420, 46.1064758
5: -21.4874229, 19.3349056, -21.4894619, 19.3367386, -40.8241577, 40.8243599
6: -22.8137627, 20.7293110, -22.8159809, 20.7313824, -43.5451393, 43.5452919
7: -27.3187599, 19.9542923, -27.3217888, 19.9563923, -47.2751541, 47.2760811
8: -33.0624084, 16.7163353, -33.0651932, 16.7184811, -49.7808876, 49.7815247
9: -19.9594021, 22.4274120, -19.9618492, 22.4300671, -42.3894653, 42.3892593

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4994619, upper bound: 27.4902497
time: 8.82 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189
time: 7.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 25.17 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.4998983, upper bound: 27.4910642
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.4992984, upper bound: 27.4910322
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.4987598, upper bound: 27.4912247
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.5000247, upper bound: 27.4913001
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.4994619, upper bound: 27.4902497
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.4994619, upper bound: 27.4902497
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 25.17
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -19.8979111, 17.9524784, -20.5171204, 18.4964161, -38.3943253, 38.4695969
1: -19.4244080, 12.3375797, -20.0016956, 12.7067089, -32.1311188, 32.3392754
2: -23.3064766, 15.3246708, -24.0310040, 15.7807617, -39.0872383, 39.3556747
3: -27.6924515, 13.3876295, -28.5493984, 13.7832432, -41.4756927, 41.9370270
4: -25.2550812, 16.2808075, -26.0280247, 16.7749100, -42.0299911, 42.3088303
5: -19.3500652, 17.4386616, -19.9456635, 17.9661674, -37.3162308, 37.3843231
6: -20.5527000, 18.6882687, -21.1813908, 19.2572823, -39.8099785, 39.8696594
7: -24.6212635, 18.0149937, -25.3709717, 18.5557537, -43.1770172, 43.3859596
8: -29.8141975, 15.0819817, -30.7130451, 15.5404806, -45.3546791, 45.7950172
9: -17.9700813, 20.2253723, -18.5238876, 20.8367329, -38.8068161, 38.7492599

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910477
time: 10.10 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910477
time: 10.22 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -21.0247726, 18.8925800, -20.4055710, 18.3981228, -39.4228973, 39.2981453
1: -20.4525108, 13.1195717, -19.8985004, 12.6413260, -33.0938339, 33.0180740
2: -24.6059265, 16.1573048, -23.9008522, 15.6990557, -40.3049812, 40.0581589
3: -29.0928154, 14.1195412, -28.3942890, 13.7118921, -42.8047028, 42.5138283
4: -26.5437832, 17.2253780, -25.8882809, 16.6862164, -43.2299995, 43.1136589
5: -20.4046879, 18.3449478, -19.8382759, 17.8710384, -38.2757263, 38.1832199
6: -21.6313629, 19.7318993, -21.0679207, 19.1547813, -40.7861404, 40.7998123
7: -25.8458099, 18.9942245, -25.2357712, 18.4584541, -44.3042564, 44.2299919
8: -31.3256435, 16.0165653, -30.5503597, 15.4587536, -46.7843933, 46.5669174
9: -18.9972782, 21.2930737, -18.4242592, 20.7263927, -39.7236710, 39.7173309

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of NS_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910475
time: 12.62 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910475
time: 9.83 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -21.6965981, 19.5364609, -20.4125118, 18.4050140, -40.1016121, 39.9489746
1: -21.0884933, 13.3762255, -19.9054832, 12.6415100, -33.7300034, 33.2817078
2: -25.4026623, 16.6332340, -23.9087524, 15.7018013, -41.1044617, 40.5419807
3: -30.2031631, 14.5329933, -28.4069405, 13.7150040, -43.9181633, 42.9399338
4: -27.5213432, 17.7113457, -25.8984604, 16.6896286, -44.2109718, 43.6098022
5: -21.0787086, 18.9775829, -19.8456039, 17.8773079, -38.9560165, 38.8231888
6: -22.3838902, 20.3362160, -21.0749550, 19.1604652, -41.5443497, 41.4111710
7: -26.8183937, 19.5801926, -25.2469597, 18.4636402, -45.2820320, 44.8271523
8: -32.4562073, 16.3874283, -30.5613899, 15.4600706, -47.9162788, 46.9488182
9: -19.5761681, 22.0104790, -18.4297085, 20.7339573, -40.3101196, 40.4401855

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4978263, upper bound: 27.4911676
time: 6.59 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4977473, upper bound: 27.4909290
time: 9.22 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -21.7558784, 19.5863914, -20.6539154, 18.6167583, -40.3726349, 40.2403069
1: -21.1425953, 13.4208460, -20.1292114, 12.7881145, -33.9307060, 33.5500565
2: -25.4714947, 16.6799240, -24.1911316, 15.8812828, -41.3527718, 40.8710518
3: -30.2760315, 14.5715389, -28.7387085, 13.8707218, -44.1467514, 43.3102417
4: -27.5905247, 17.7618618, -26.1988125, 16.8839245, -44.4744492, 43.9606743
5: -21.1352577, 19.0241737, -20.0777664, 18.0824261, -39.2176819, 39.1019363
6: -22.4407349, 20.3916550, -21.3201542, 19.3830338, -41.8237686, 41.7118073
7: -26.8820210, 19.6319008, -25.5366669, 18.6752720, -45.5572929, 45.1685562
8: -32.5321770, 16.4379311, -30.9116974, 15.6415634, -48.1737404, 47.3496284
9: -19.6294689, 22.0662918, -18.6465092, 20.9719963, -40.6014633, 40.7127991

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4993324, upper bound: 27.4912618
time: 7.10 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4992984, upper bound: 27.4910325
time: 11.21 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -19.8979111, 17.9524784, -21.9643784, 19.7696304, -39.6675377, 39.9168549
1: -19.4244080, 12.3375797, -21.3380318, 13.5551529, -32.9795570, 33.6756134
2: -23.3064766, 15.3246708, -25.7162876, 16.8387451, -40.1452141, 41.0409584
3: -27.6924515, 13.3876295, -30.5611649, 14.7090855, -42.4015350, 43.9487953
4: -25.2550812, 16.2808075, -27.8500500, 17.9334717, -43.1885529, 44.1308594
5: -19.3500652, 17.4386616, -21.3360310, 19.2022095, -38.5522690, 38.7746887
6: -20.5527000, 18.6882687, -22.6549053, 20.5853596, -41.1380615, 41.3431740
7: -24.6212635, 18.0149937, -27.1300735, 19.8173256, -44.4385910, 45.1450653
8: -29.8141975, 15.0819817, -32.8349037, 16.6005936, -46.4147835, 47.9168777
9: -17.9700813, 20.2253723, -19.8190575, 22.2730446, -40.2431259, 40.0444298

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 119

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
time: 7.76 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
time: 23.57 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -21.0247726, 18.8925800, -21.8535881, 19.6718464, -40.6966171, 40.7461624
1: -20.4525108, 13.1195717, -21.2354736, 13.4894495, -33.9419594, 34.3550453
2: -24.6059265, 16.1573048, -25.5868244, 16.7573643, -41.3632889, 41.7441292
3: -29.0928154, 14.1195412, -30.4070816, 14.6379986, -43.7308121, 44.5266190
4: -26.5437832, 17.2253780, -27.7110920, 17.8451767, -44.3889580, 44.9364700
5: -20.4046879, 18.3449478, -21.2289696, 19.1079102, -39.5125961, 39.5739136
6: -21.6313629, 19.7318993, -22.5420132, 20.4834175, -42.1147804, 42.2739105
7: -25.8458099, 18.9942245, -26.9956608, 19.7203064, -45.5661163, 45.9898834
8: -31.3256435, 16.0165653, -32.6732407, 16.5191498, -47.8447876, 48.6898003
9: -18.9972782, 21.2930737, -19.7195644, 22.1631927, -41.1604691, 41.0126343

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
time: 10.60 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
time: 9.04 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -21.3367233, 19.2174301, -21.9643784, 19.7696304, -41.1063538, 41.1818085
1: -20.7524586, 13.1773529, -21.3380318, 13.5551529, -34.3076057, 34.5153809
2: -24.9810219, 16.3758373, -25.7162876, 16.8387451, -41.8197670, 42.0921249
3: -29.6928291, 14.3067036, -30.5611649, 14.7090855, -44.4019165, 44.8678665
4: -27.0660973, 17.4313679, -27.8500500, 17.9334717, -44.9995613, 45.2814178
5: -20.7291203, 18.6693287, -21.3360310, 19.2022095, -39.9313278, 40.0053596
6: -22.0174046, 20.0076370, -22.6549053, 20.5853596, -42.6027603, 42.6625443
7: -26.3708858, 19.2671032, -27.1300735, 19.8173256, -46.1882095, 46.3971786
8: -31.9239502, 16.1345272, -32.8349037, 16.6005936, -48.5245399, 48.9694290
9: -19.2548180, 21.6518669, -19.8190575, 22.2730446, -41.5278625, 41.4709244

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189
time: 5.84 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189
time: 12.62 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -22.4598789, 20.1572170, -21.8535881, 19.6718464, -42.1317253, 42.0108032
1: -21.7780991, 13.9628868, -21.2354736, 13.4894495, -35.2675476, 35.1983604
2: -26.2801132, 17.2063389, -25.5868244, 16.7573643, -43.0374756, 42.7931633
3: -31.0875015, 15.0386696, -30.4070816, 14.6379986, -45.7254944, 45.4457512
4: -28.3524017, 18.3756332, -27.7110920, 17.8451767, -46.1975784, 46.0867195
5: -21.7859936, 19.5700359, -21.2289696, 19.1079102, -40.8939056, 40.7990036
6: -23.0947132, 21.0489521, -22.5420132, 20.4834175, -43.5781326, 43.5909653
7: -27.5908356, 20.2454300, -26.9956608, 19.7203064, -47.3111420, 47.2410889
8: -33.4308243, 17.0679893, -32.6732407, 16.5191498, -49.9499702, 49.7412300
9: -20.2849560, 22.7169266, -19.7195644, 22.1631927, -42.4481506, 42.4364853

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4934102, upper bound: 27.4892657
time: 11.32 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4983784, upper bound: 27.4896664
time: 10.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 22.46 seconds
NS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910477
NS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910477
NS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910475
NS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4994999, upper bound: 27.4910475
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4978263, upper bound: 27.4911676
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4977473, upper bound: 27.4909290
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4993324, upper bound: 27.4912618
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4992984, upper bound: 27.4910325
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4989480, upper bound: 27.4902254
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4988592, upper bound: 27.4902189
NS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4934102, upper bound: 27.4892657
NS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.46
Output dim: 1, lower bound: -27.4983784, upper bound: 27.4896664

## BFS NS instance: NS_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -19.8979111, 17.9524784, -19.8987808, 17.9532013, -37.8511086, 37.8512573
1: -19.4244080, 12.3375797, -19.4252090, 12.3381786, -31.7625866, 31.7627792
2: -23.3064766, 15.3246708, -23.3074913, 15.3253222, -38.6317978, 38.6321602
3: -27.6924515, 13.3876295, -27.6935444, 13.3881989, -41.0806503, 41.0811729
4: -25.2550812, 16.2808075, -25.2560806, 16.2815304, -41.5366135, 41.5368805
5: -19.3500652, 17.4386616, -19.3508778, 17.4393692, -36.7894363, 36.7895393
6: -20.5527000, 18.6882687, -20.5535297, 18.6890774, -39.2417755, 39.2417984
7: -24.6212635, 18.0149937, -24.6222286, 18.0157585, -42.6370239, 42.6372223
8: -29.8141975, 15.0819817, -29.8153648, 15.0826988, -44.8968887, 44.8973389
9: -17.9700813, 20.2253723, -17.9708748, 20.2262115, -38.1962929, 38.1962471

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 197

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 119

### Candidate
type: A, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 208

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4950179, upper bound: 27.4892581
time: 12.88 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

### Candidate
type: B, layer: 1, pos: 234

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 234

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 133

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 221

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4951979, upper bound: 27.4907165
time: 8.04 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 198

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 156

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

### Candidate
type: B, layer: 1, pos: 162

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 162

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 56

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 59

## Relational analysis of NS_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 59

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4998075, upper bound: 27.4910610
time: 8.93 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4993990, upper bound: 27.4902616
time: 7.35 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -19.8979111, 17.9524784, -21.0085831, 18.8799400, -38.7778511, 38.9610519
1: -19.4244080, 12.3375797, -20.4402657, 13.1070166, -32.5314255, 32.7778473
2: -23.3064766, 15.3246708, -24.5875835, 16.1443920, -39.4508667, 39.9122543
3: -27.6924515, 13.3876295, -29.0713673, 14.1069899, -41.7994423, 42.4589958
4: -25.2550812, 16.2808075, -26.5236645, 17.2098103, -42.4648895, 42.8044662
5: -19.3500652, 17.4386616, -20.3810043, 18.3370934, -37.6871567, 37.8196602
6: -20.5527000, 18.6882687, -21.6147652, 19.7156715, -40.2683678, 40.3030319
7: -24.6212635, 18.0149937, -25.8288097, 18.9745216, -43.5957870, 43.8437996
8: -29.8141975, 15.0819817, -31.3026314, 16.0037956, -45.8179932, 46.3846054
9: -17.9700813, 20.2253723, -18.9766006, 21.2753487, -39.2454300, 39.2019730

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 184

### Candidate
type: A, layer: 1, pos: 127

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 184

### Candidate
type: B, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 250

### Candidate
type: B, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 215

### Candidate
type: B, layer: 1, pos: 178

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 120

### Candidate
type: A, layer: 1, pos: 144

### Candidate
type: B, layer: 1, pos: 119

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 94

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4950179, upper bound: 27.4892581
time: 7.10 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 178

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 208

## Relational analysis of NS_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 208

### Candidate
type: B, layer: 1, pos: 234

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of NS_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4951979, upper bound: 27.4907165
time: 5.96 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 234

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 56

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 59

## Relational analysis of NS_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 16.18 + 585.31 = 601.50 seconds
