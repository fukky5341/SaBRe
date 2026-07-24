## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 105.986448459


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352)
1: (-48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901)
2: (-63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453)
3: (-66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463)
4: (-61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134)
5: (-55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531)
6: (-53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280)
7: (-57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380)
8: (-70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773)
9: (-53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.80 + 11.10 = 11.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -106.0925410, upper bound: 106.0925410

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0728111, upper bound: 106.0743535
time: 8.85 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0692145, upper bound: 106.0692145
time: 5.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 14.85
Output dim: 7, lower bound: -106.0728111, upper bound: 106.0743535
NS_A2, status: Status.UNKNOWN, split count: 1, time: 14.85
Output dim: 7, lower bound: -106.0692145, upper bound: 106.0692145

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -57.1603050, 45.8663750, -57.8715248, 46.4339218, -103.5942001, 103.7378998
1: -47.6767387, 40.5406494, -48.2762108, 41.0486870, -88.7254257, 88.8168640
2: -63.1810570, 41.1228867, -63.9665565, 41.6315002, -104.8125610, 105.0894318
3: -66.0987549, 35.7515068, -66.9424515, 36.1956024, -102.2943573, 102.6939392
4: -60.9447250, 46.7464638, -61.7093658, 47.3356514, -108.2803726, 108.4558258
5: -54.9421501, 42.7683449, -55.6309967, 43.3028641, -98.2450104, 98.3993225
6: -52.4629593, 51.0089493, -53.1199455, 51.6415939, -104.1045532, 104.1288834
7: -56.9693298, 48.6926842, -57.6861839, 49.2933502, -106.2626801, 106.3788605
8: -70.0121231, 48.3054657, -70.8696365, 48.8952446, -118.9073639, 119.1750946
9: -52.5454826, 51.5988922, -53.1959190, 52.2425995, -104.7880554, 104.7948074

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

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
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0676394, upper bound: 106.0692917
time: 7.56 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0679242, upper bound: 106.0694938
time: 7.41 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -50.3040352, 40.4271278, -56.1227722, 45.0383797, -95.3424072, 96.5498962
1: -41.8306999, 35.6165237, -46.8023071, 39.8002167, -81.6309128, 82.4188309
2: -55.6620064, 36.2933617, -62.0342636, 40.3837395, -96.0457458, 98.3276215
3: -58.0784683, 31.4425945, -64.8729324, 35.1081085, -93.1865768, 96.3155136
4: -53.6313629, 41.0127983, -59.8287506, 45.8877716, -99.5191345, 100.8415298
5: -48.1832848, 37.5342979, -53.9328728, 41.9890709, -90.1723404, 91.4671555
6: -46.1856880, 44.9543228, -51.5083351, 50.0885315, -96.2742081, 96.4626389
7: -50.1178131, 42.8803482, -55.9277382, 47.8148079, -97.9326172, 98.8080750
8: -61.8760452, 42.5751419, -68.7657700, 47.4458237, -109.3218689, 111.3409042
9: -46.2887001, 45.3772888, -51.5958900, 50.6590729, -96.9477692, 96.9731445

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 188

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0638486, upper bound: 106.0638114
time: 6.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0641374, upper bound: 106.0641374
time: 5.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.06 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 7, lower bound: -106.0676394, upper bound: 106.0692917
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 7, lower bound: -106.0679242, upper bound: 106.0694938
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 7, lower bound: -106.0638486, upper bound: 106.0638114
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.06
Output dim: 7, lower bound: -106.0641374, upper bound: 106.0641374

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -53.8999481, 43.2589340, -45.3923492, 36.4707603, -90.3707123, 88.6512833
1: -44.9181175, 38.2087097, -37.6797333, 32.1286964, -77.0468140, 75.8884430
2: -59.5410576, 38.7582550, -50.0246849, 32.6191750, -92.1602325, 88.7829437
3: -62.2400856, 33.7395096, -52.1956406, 28.4680977, -90.7081833, 85.9351501
4: -57.4103355, 44.0461769, -48.1822968, 37.0192795, -94.4296036, 92.2284698
5: -51.8051224, 40.3361588, -43.6188278, 33.9815331, -85.7866516, 83.9549789
6: -49.4232445, 48.1126213, -41.4607735, 40.5465126, -89.9697571, 89.5733948
7: -53.6464424, 45.9135132, -44.9528160, 38.6325073, -92.2789459, 90.8663254
8: -66.0190353, 45.6175499, -55.5584183, 38.5967026, -104.6157303, 101.1759567
9: -49.5334778, 48.6053200, -41.6406708, 40.7653160, -90.2987976, 90.2459793

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672601, upper bound: 106.0688566
time: 8.16 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672601, upper bound: 106.0692917
time: 8.05 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -54.3103409, 43.5897865, -49.8226051, 40.0181923, -94.3285294, 93.4123840
1: -45.2720299, 38.5102959, -41.4727058, 35.3105278, -80.5825577, 79.9830017
2: -60.0111351, 39.0611267, -55.0167274, 35.8127441, -95.8238678, 94.0778503
3: -62.7303772, 33.9961853, -57.4373703, 31.2368050, -93.9671783, 91.4335556
4: -57.8594894, 44.3862762, -53.0002327, 40.6760025, -98.5354919, 97.3865051
5: -52.1985550, 40.6437492, -47.8841286, 37.3114433, -89.5099792, 88.5278702
6: -49.8138580, 48.4812889, -45.6264381, 44.5071449, -94.3209991, 94.1077271
7: -54.0777397, 46.2680931, -49.5189590, 42.4413567, -96.5190964, 95.7870483
8: -66.5420074, 45.9553146, -61.0541229, 42.2546043, -108.7966003, 107.0094376
9: -49.9205132, 48.9967499, -45.7715416, 44.8748474, -94.7953644, 94.7682953

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0673208, upper bound: 106.0688757
time: 9.32 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0673208, upper bound: 106.0694938
time: 9.40 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -47.2853889, 38.0245438, -43.7798615, 35.1924210, -82.4778137, 81.8044052
1: -39.2565651, 33.4564552, -36.3115807, 30.9778728, -70.2344360, 69.7680359
2: -52.2947006, 34.1230736, -48.2468452, 31.4809017, -83.7756042, 82.3699188
3: -54.5213165, 29.5640736, -50.3006706, 27.4582882, -81.9796066, 79.8647385
4: -50.3564034, 38.5249214, -46.4463768, 35.6931267, -86.0495071, 84.9712982
5: -45.2865143, 35.2722054, -42.0548706, 32.7652893, -78.0517960, 77.3270721
6: -43.3663712, 42.2648773, -39.9689140, 39.1129265, -82.4792938, 82.2337875
7: -47.0323372, 40.2946854, -43.3317032, 37.2628899, -84.2952271, 83.6263885
8: -58.1645622, 40.0873833, -53.6132393, 37.2584763, -95.4230347, 93.7006149
9: -43.4858017, 42.6004105, -40.1579590, 39.3029709, -82.7887726, 82.7583694

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0637112, upper bound: 106.0637112
time: 7.27 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0637112, upper bound: 106.0638114
time: 6.92 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -47.6205368, 38.2958755, -48.1660194, 38.7031822, -86.3237152, 86.4618988
1: -39.5514603, 33.7043228, -40.0687370, 34.1293983, -73.6808548, 73.7730484
2: -52.6831207, 34.3701019, -53.1867485, 34.6388054, -87.3219299, 87.5568390
3: -54.9215012, 29.7750320, -55.4912453, 30.1993675, -85.1208572, 85.2662811
4: -50.7231712, 38.8043060, -51.2180023, 39.3147583, -90.0379257, 90.0223083
5: -45.6075172, 35.5235596, -46.2755318, 36.0636597, -81.6711731, 81.7990875
6: -43.6872406, 42.5701447, -44.0963707, 43.0366135, -86.7238388, 86.6665115
7: -47.3903580, 40.5859184, -47.8543282, 41.0330963, -88.4234467, 88.4402390
8: -58.5982933, 40.3650284, -59.0545807, 40.8813362, -99.4796219, 99.4196091
9: -43.8053894, 42.9242592, -44.2508850, 43.3713799, -87.1767654, 87.1751404

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 188

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0638114, upper bound: 106.0638486
time: 7.23 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0638114, upper bound: 106.0641374
time: 6.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.86 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0672601, upper bound: 106.0688566
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0672601, upper bound: 106.0692917
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0673208, upper bound: 106.0688757
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0673208, upper bound: 106.0694938
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0637112, upper bound: 106.0637112
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0637112, upper bound: 106.0638114
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0638114, upper bound: 106.0638486
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.86
Output dim: 7, lower bound: -106.0638114, upper bound: 106.0641374

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -44.7261047, 35.9428711, -45.3923492, 36.4707603, -81.1968689, 81.3352127
1: -37.1146851, 31.6530685, -37.6797333, 32.1286964, -69.2433701, 69.3328018
2: -49.2908821, 32.1479149, -50.0246849, 32.6191750, -81.9100571, 82.1725998
3: -51.4098625, 28.0487537, -52.1956406, 28.4680977, -79.8779602, 80.2443848
4: -47.4652405, 36.4712257, -48.1822968, 37.0192795, -84.4845123, 84.6535187
5: -42.9740982, 33.4789276, -43.6188278, 33.9815331, -76.9556274, 77.0977325
6: -40.8428307, 39.9532433, -41.4607735, 40.5465126, -81.3893433, 81.4139938
7: -44.2817574, 38.0670700, -44.9528160, 38.6325073, -82.9142609, 83.0198822
8: -54.7531586, 38.0439148, -55.5584183, 38.5967026, -93.3498459, 93.6023254
9: -41.0288315, 40.1615562, -41.6406708, 40.7653160, -81.7941437, 81.8022308

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0555627, upper bound: 106.0565281
time: 8.23 seconds

## Relational analysis of NS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 177

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0644852, upper bound: 106.0655268
time: 8.86 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643035, upper bound: 106.0654880
time: 8.39 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -49.1475410, 39.4817505, -45.3923492, 36.4707603, -85.6183014, 84.8740921
1: -40.9007683, 34.8288651, -37.6797333, 32.1286964, -73.0294495, 72.5085983
2: -54.2703514, 35.3330193, -50.0246849, 32.6191750, -86.8895264, 85.3577042
3: -56.6398277, 30.8125000, -52.1956406, 28.4680977, -85.1079254, 83.0081406
4: -52.2740250, 40.1206703, -48.1822968, 37.0192795, -89.2933044, 88.3029633
5: -47.2278595, 36.8040390, -43.6188278, 33.9815331, -81.2093887, 80.4228516
6: -45.0011597, 43.9066200, -41.4607735, 40.5465126, -85.5476685, 85.3673935
7: -48.8386726, 41.8680687, -44.9528160, 38.6325073, -87.4711609, 86.8208847
8: -60.2361183, 41.6962090, -55.5584183, 38.5967026, -98.8328247, 97.2546234
9: -45.1528587, 44.2615356, -41.6406708, 40.7653160, -85.9181747, 85.9022064

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0654296, upper bound: 106.0673324
time: 10.34 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643035, upper bound: 106.0657375
time: 10.27 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -44.7261047, 35.9428711, -49.8226051, 40.0181923, -84.7442932, 85.7654724
1: -37.1146851, 31.6530685, -41.4727058, 35.3105278, -72.4252014, 73.1257782
2: -49.2908821, 32.1479149, -55.0167274, 35.8127441, -85.1036224, 87.1646423
3: -51.4098625, 28.0487537, -57.4373703, 31.2368050, -82.6466522, 85.4861145
4: -47.4652405, 36.4712257, -53.0002327, 40.6760025, -88.1412354, 89.4714584
5: -42.9740982, 33.4789276, -47.8841286, 37.3114433, -80.2855377, 81.3630447
6: -40.8428307, 39.9532433, -45.6264381, 44.5071449, -85.3499756, 85.5796585
7: -44.2817574, 38.0670700, -49.5189590, 42.4413567, -86.7231140, 87.5860291
8: -54.7531586, 38.0439148, -61.0541229, 42.2546043, -97.0077438, 99.0980377
9: -41.0288315, 40.1615562, -45.7715416, 44.8748474, -85.9036789, 85.9330978

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0552595, upper bound: 106.0561466
time: 9.12 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643996, upper bound: 106.0654584
time: 8.35 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0641733, upper bound: 106.0654010
time: 8.68 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -49.1475410, 39.4817505, -49.8226051, 40.0181923, -89.1657333, 89.3043518
1: -40.9007683, 34.8288651, -41.4727058, 35.3105278, -76.2112885, 76.3015747
2: -54.2703514, 35.3330193, -55.0167274, 35.8127441, -90.0830841, 90.3497467
3: -56.6398277, 30.8125000, -57.4373703, 31.2368050, -87.8766174, 88.2498703
4: -52.2740250, 40.1206703, -53.0002327, 40.6760025, -92.9500275, 93.1209030
5: -47.2278595, 36.8040390, -47.8841286, 37.3114433, -84.5392838, 84.6881714
6: -45.0011597, 43.9066200, -45.6264381, 44.5071449, -89.5083008, 89.5330582
7: -48.8386726, 41.8680687, -49.5189590, 42.4413567, -91.2800140, 91.3870239
8: -60.2361183, 41.6962090, -61.0541229, 42.2546043, -102.4907227, 102.7503357
9: -45.1528587, 44.2615356, -45.7715416, 44.8748474, -90.0277023, 90.0330811

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0552595, upper bound: 106.0569552
time: 8.80 seconds

## Relational analysis of NS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 177

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643996, upper bound: 106.0660017
time: 9.06 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0641733, upper bound: 106.0659643
time: 10.22 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -38.7531967, 31.1966152, -43.7798615, 35.1924210, -73.9456177, 74.9764709
1: -31.9917488, 27.3697948, -36.3115807, 30.9778728, -62.9696198, 63.6813736
2: -42.7588348, 27.9836159, -48.2468452, 31.4809017, -74.2397385, 76.2304611
3: -44.4469910, 24.2342186, -50.3006706, 27.4582882, -71.9052811, 74.5348740
4: -41.1044884, 31.4857750, -46.4463768, 35.6931267, -76.7976151, 77.9321518
5: -37.0884972, 28.8538799, -42.0548706, 32.7652893, -69.8537903, 70.9087524
6: -35.3680649, 34.6551552, -39.9689140, 39.1129265, -74.4809875, 74.6240616
7: -38.2861633, 32.9730225, -43.3317032, 37.2628899, -75.5490570, 76.3047256
8: -47.6487236, 33.0844193, -53.6132393, 37.2584763, -84.9071960, 86.6976547
9: -35.5454865, 34.7291336, -40.1579590, 39.3029709, -74.8484573, 74.8870850

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 188

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0616929, upper bound: 106.0615871
time: 6.86 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0613461, upper bound: 106.0613266
time: 6.42 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -42.6414642, 34.3307419, -43.7798615, 35.1924210, -77.8338852, 78.1106033
1: -35.3151169, 30.1513462, -36.3115807, 30.9778728, -66.2929764, 66.4629288
2: -47.1402435, 30.7896843, -48.2468452, 31.4809017, -78.6211472, 79.0365295
3: -49.0475845, 26.6716557, -50.3006706, 27.4582882, -76.5058746, 76.9723206
4: -45.3253593, 34.7044716, -46.4463768, 35.6931267, -81.0184784, 81.1508484
5: -40.8263092, 31.7912598, -42.0548706, 32.7652893, -73.5915909, 73.8461227
6: -39.0295372, 38.1339874, -39.9689140, 39.1129265, -78.1424637, 78.1029053
7: -42.3141975, 36.3181610, -43.3317032, 37.2628899, -79.5770874, 79.6498642
8: -52.4899597, 36.2730598, -53.6132393, 37.2584763, -89.7484360, 89.8862991
9: -39.1881752, 38.3465996, -40.1579590, 39.3029709, -78.4911499, 78.5045624

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 92

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0616929, upper bound: 106.0616608
time: 7.11 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0613461, upper bound: 106.0613924
time: 7.78 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -38.7531967, 31.1966152, -48.1660194, 38.7031822, -77.4563751, 79.3626328
1: -31.9917488, 27.3697948, -40.0687370, 34.1293983, -66.1211472, 67.4385300
2: -42.7588348, 27.9836159, -53.1867485, 34.6388054, -77.3976440, 81.1703644
3: -44.4469910, 24.2342186, -55.4912453, 30.1993675, -74.6463623, 79.7254639
4: -41.1044884, 31.4857750, -51.2180023, 39.3147583, -80.4192429, 82.7037735
5: -37.0884972, 28.8538799, -46.2755318, 36.0636597, -73.1521606, 75.1294098
6: -35.3680649, 34.6551552, -44.0963707, 43.0366135, -78.4046707, 78.7515259
7: -38.2861633, 32.9730225, -47.8543282, 41.0330963, -79.3192596, 80.8273468
8: -47.6487236, 33.0844193, -59.0545807, 40.8813362, -88.5300598, 92.1389999
9: -35.5454865, 34.7291336, -44.2508850, 43.3713799, -78.9168549, 78.9800110

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 221

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0616688, upper bound: 106.0617207
time: 7.07 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0612997, upper bound: 106.0613940
time: 6.98 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.6414642, 34.3307419, -48.1660194, 38.7031822, -81.3446503, 82.4967651
1: -35.3151169, 30.1513462, -40.0687370, 34.1293983, -69.4445114, 70.2200851
2: -47.1402435, 30.7896843, -53.1867485, 34.6388054, -81.7790527, 83.9764252
3: -49.0475845, 26.6716557, -55.4912453, 30.1993675, -79.2469482, 82.1629028
4: -45.3253593, 34.7044716, -51.2180023, 39.3147583, -84.6400986, 85.9224701
5: -40.8263092, 31.7912598, -46.2755318, 36.0636597, -76.8899612, 78.0667877
6: -39.0295372, 38.1339874, -44.0963707, 43.0366135, -82.0661469, 82.2303619
7: -42.3141975, 36.3181610, -47.8543282, 41.0330963, -83.3472824, 84.1724777
8: -52.4899597, 36.2730598, -59.0545807, 40.8813362, -93.3712921, 95.3276367
9: -39.1881752, 38.3465996, -44.2508850, 43.3713799, -82.5595474, 82.5974884

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 221

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0616688, upper bound: 106.0619950
time: 7.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0612997, upper bound: 106.0617243
time: 6.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 24.58 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0644852, upper bound: 106.0655268
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0643035, upper bound: 106.0654880
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0654296, upper bound: 106.0673324
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0643035, upper bound: 106.0657375
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0643996, upper bound: 106.0654584
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0641733, upper bound: 106.0654010
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0643996, upper bound: 106.0660017
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0641733, upper bound: 106.0659643
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0616929, upper bound: 106.0615871
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0613461, upper bound: 106.0613266
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0616929, upper bound: 106.0616608
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0613461, upper bound: 106.0613924
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0616688, upper bound: 106.0617207
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0612997, upper bound: 106.0613940
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0616688, upper bound: 106.0619950
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.58
Output dim: 7, lower bound: -106.0612997, upper bound: 106.0617243

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -44.6298103, 35.8661461, -43.7533760, 35.1634750, -79.7932587, 79.6195221
1: -37.0335808, 31.5855865, -36.2984886, 30.9793625, -68.0129395, 67.8840790
2: -49.1843796, 32.0809555, -48.2119865, 31.4772606, -80.6616364, 80.2929382
3: -51.2961388, 27.9894676, -50.2589645, 27.4580746, -78.7542038, 78.2484283
4: -47.3609276, 36.3921814, -46.4062920, 35.6742554, -83.0351715, 82.7984543
5: -42.8808098, 33.4072571, -42.0309868, 32.7594681, -75.6402740, 75.4382477
6: -40.7531624, 39.8672485, -39.9338226, 39.0834045, -79.8365631, 79.8010712
7: -44.1854210, 37.9865074, -43.3118401, 37.2604599, -81.4458771, 81.2983475
8: -54.6364632, 37.9633026, -53.5704765, 37.2242737, -91.8607330, 91.5337677
9: -40.9405785, 40.0748672, -40.1378212, 39.2890472, -80.2296295, 80.2126923

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643616, upper bound: 106.0655540
time: 9.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643616, upper bound: 106.0655540
time: 9.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -43.4930496, 34.9563141, -45.1575737, 36.2309837, -79.7240295, 80.1138916
1: -36.0772400, 30.7890873, -37.4813957, 31.9835491, -68.0607910, 68.2704849
2: -47.9288712, 31.2912216, -49.7777290, 32.4999962, -80.4288635, 81.0689392
3: -49.9533691, 27.2911282, -51.9040184, 28.3106899, -78.2640533, 79.1951370
4: -46.1251717, 35.4565468, -47.8859253, 36.7834892, -82.9086609, 83.3424683
5: -41.7774429, 32.5568047, -43.3412399, 33.7779961, -75.5554352, 75.8980408
6: -39.6907196, 38.8541374, -41.2063103, 40.3117218, -80.0024338, 80.0604324
7: -43.0457382, 37.0385437, -44.7076836, 38.4766998, -81.5224380, 81.7462311
8: -53.2598839, 37.0085640, -55.2770538, 38.3070183, -91.5669022, 92.2856064
9: -39.8993187, 39.0521202, -41.4280052, 40.5651817, -80.4644928, 80.4801254

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 92

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643616, upper bound: 106.0655540
time: 8.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0643616, upper bound: 106.0655540
time: 8.20 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -47.4950218, 38.1641998, -45.2956009, 36.3936729, -83.8886948, 83.4597931
1: -39.5092087, 33.6704903, -37.5982552, 32.0608940, -71.5700912, 71.2687454
2: -52.4407654, 34.1822586, -49.9176750, 32.5518951, -84.9926605, 84.0999298
3: -54.6900978, 29.7940979, -52.0813484, 28.4085178, -83.0986099, 81.8754425
4: -50.4827690, 38.7639847, -48.0774612, 36.9398613, -87.4226303, 86.8414459
5: -45.6278725, 35.5729980, -43.5250969, 33.9095192, -79.5373840, 79.0980911
6: -43.4633980, 42.4317970, -41.3706741, 40.4601173, -83.9235153, 83.8024673
7: -47.1854630, 40.4849205, -44.8560143, 38.5515556, -85.7370148, 85.3409348
8: -58.2336540, 40.3145409, -55.4411812, 38.5157356, -96.7493896, 95.7557144
9: -43.6383362, 42.7728233, -41.5520058, 40.6782227, -84.3165512, 84.3248291

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0645026, upper bound: 106.0657375
time: 11.01 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0645026, upper bound: 106.0657375
time: 9.43 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -48.8425980, 39.1915054, -44.1585312, 35.4845657, -84.3271637, 83.3500137
1: -40.6446114, 34.6352234, -36.6421242, 31.2647438, -71.9093475, 71.2773438
2: -53.9471130, 35.1689758, -48.6619911, 31.7632694, -85.7103806, 83.8309631
3: -56.2697296, 30.6144028, -50.7387352, 27.7105103, -83.9802399, 81.3531342
4: -51.9007416, 39.8253822, -46.8417740, 36.0039330, -87.9046555, 86.6671600
5: -46.8869781, 36.5494423, -42.4215508, 33.0600853, -79.9470673, 78.9709930
6: -44.6843224, 43.6097145, -40.3086624, 39.4466209, -84.1309280, 83.9183807
7: -48.5274277, 41.6558456, -43.7169456, 37.6037064, -86.1311340, 85.3727875
8: -59.8776970, 41.3455505, -54.0654335, 37.5611420, -97.4388199, 95.4109726
9: -44.8780174, 44.0020714, -40.5109749, 39.6557388, -84.5337524, 84.5130386

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 92

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0645026, upper bound: 106.0657375
time: 10.69 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0645026, upper bound: 106.0657375
time: 8.88 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -44.6298103, 35.8661461, -48.1517029, 38.6857834, -83.3155823, 84.0178528
1: -37.0335808, 31.5855865, -40.0664978, 34.1398315, -71.1734161, 71.6520844
2: -49.1843796, 32.0809555, -53.1668587, 34.6490555, -83.8334351, 85.2478104
3: -51.2961388, 27.9894676, -55.4639931, 30.2079792, -81.5041199, 83.4534607
4: -47.3609276, 36.3921814, -51.1894875, 39.3049622, -86.6658783, 87.5816574
5: -42.8808098, 33.4072571, -46.2635994, 36.0687294, -78.9495392, 79.6708527
6: -40.7531624, 39.8672485, -44.0727043, 43.0163956, -83.7695618, 83.9399567
7: -44.1854210, 37.9865074, -47.8471909, 41.0435410, -85.2289581, 85.8336945
8: -54.6364632, 37.9633026, -59.0282631, 40.8596458, -95.4961090, 96.9915619
9: -40.9405785, 40.0748672, -44.2418900, 43.3692436, -84.3098221, 84.3167572

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0642194, upper bound: 106.0654010
time: 8.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0642194, upper bound: 106.0654010
time: 9.28 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -43.4930496, 34.9563141, -49.5094490, 39.7253799, -83.2184296, 84.4657593
1: -36.0772400, 30.7890873, -41.2133446, 35.1136703, -71.1909103, 72.0024261
2: -47.9288712, 31.2912216, -54.6858292, 35.6474304, -83.5762939, 85.9770508
3: -49.9533691, 27.2911282, -57.0595093, 31.0350685, -80.9884338, 84.3506165
4: -46.1251717, 35.4565468, -52.6190605, 40.3764687, -86.5016327, 88.0756073
5: -41.7774429, 32.5568047, -47.5384636, 37.0524712, -78.8299103, 80.0952606
6: -39.6907196, 38.8541374, -45.3085098, 44.2071037, -83.8978271, 84.1626434
7: -43.0457382, 37.0385437, -49.2011909, 42.2240562, -85.2697906, 86.2397308
8: -53.2598839, 37.0085640, -60.6870155, 41.9003296, -95.1602173, 97.6955719
9: -39.8993187, 39.0521202, -45.4911919, 44.6081543, -84.5074768, 84.5432892

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 92

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

### Candidate
type: B, layer: 1, pos: 83

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0642194, upper bound: 106.0654010
time: 8.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0642194, upper bound: 106.0654010
time: 8.55 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -49.0499916, 39.4039955, -48.1517029, 38.6857834, -87.7357635, 87.5556946
1: -40.8187218, 34.7605057, -40.0664978, 34.1398315, -74.9585571, 74.8270035
2: -54.1623459, 35.2651062, -53.1668587, 34.6490555, -88.8113861, 88.4319611
3: -56.5246162, 30.7524853, -55.4639931, 30.2079792, -86.7325974, 86.2164764
4: -52.1683083, 40.0406532, -51.1894875, 39.3049622, -91.4732666, 91.2301407
5: -47.1333008, 36.7315025, -46.2635994, 36.0687294, -83.2020264, 82.9951019
6: -44.9104385, 43.8196220, -44.0727043, 43.0163956, -87.9268341, 87.8923187
7: -48.7411232, 41.7864914, -47.8471909, 41.0435410, -89.7846680, 89.6336823
8: -60.1178741, 41.6147652, -59.0282631, 40.8596458, -100.9775085, 100.6430283
9: -45.0635643, 44.1736450, -44.2418900, 43.3692436, -88.4328079, 88.4155350

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0647499, upper bound: 106.0659643
time: 8.46 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0647499, upper bound: 106.0659643
time: 8.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.8969269, 38.4824677, -49.5094490, 39.7253799, -87.6223068, 87.9919128
1: -39.8499603, 33.9532623, -41.2133446, 35.1136703, -74.9636230, 75.1665878
2: -52.8873253, 34.4649124, -54.6858292, 35.6474304, -88.5347366, 89.1507339
3: -55.1662598, 30.0443306, -57.0595093, 31.0350685, -86.2013245, 87.1038284
4: -50.9151993, 39.0918198, -52.6190605, 40.3764687, -91.2916641, 91.7108688
5: -46.0153694, 35.8705826, -47.5384636, 37.0524712, -83.0678406, 83.4090347
6: -43.8351822, 42.7920074, -45.3085098, 44.2071037, -88.0422821, 88.1005173
7: -47.5876770, 40.8249702, -49.2011909, 42.2240562, -89.8117294, 90.0261612
8: -58.7249489, 40.6460686, -60.6870155, 41.9003296, -100.6252747, 101.3330841
9: -44.0083008, 43.1363144, -45.4911919, 44.6081543, -88.6164551, 88.6274796

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0647499, upper bound: 106.0659643
time: 10.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0647499, upper bound: 106.0659643
time: 9.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -38.6598625, 31.1218014, -42.1422195, 33.8846016, -72.5444565, 73.2640228
1: -31.9134960, 27.3047523, -34.9308624, 29.8282795, -61.7417717, 62.2356110
2: -42.6563148, 27.9187412, -46.4361649, 30.3373146, -72.9936295, 74.3549042
3: -44.3370781, 24.1762543, -48.3645477, 26.4480953, -70.7851715, 72.5407944
4: -41.0043526, 31.4094715, -44.6714516, 34.3495178, -75.3538666, 76.0809250
5: -36.9983978, 28.7839584, -40.4685402, 31.5419941, -68.5403824, 69.2525024
6: -35.2816086, 34.5722466, -38.4418030, 37.6518936, -72.9335022, 73.0140457
7: -38.1928864, 32.8951950, -41.6904449, 35.8916130, -74.0844955, 74.5856323
8: -47.5359955, 33.0079193, -51.6250305, 35.8871117, -83.4231110, 84.6329498
9: -35.4598885, 34.6456490, -38.6551285, 37.8276749, -73.2875595, 73.3007736

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 188

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 219

### Candidate
type: B, layer: 1, pos: 219

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 132

### Candidate
type: A, layer: 1, pos: 89

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 182

### Candidate
type: A, layer: 1, pos: 132

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0614764, upper bound: 106.0614764
time: 6.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0614764, upper bound: 106.0614764
time: 7.05 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 11.90 + 594.40 = 606.29 seconds
