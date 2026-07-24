## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 17.96064755562


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208)
1: (-7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301)
2: (-4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755)
3: (-8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169)
4: (-5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.05 + 1.78 = 2.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -17.9624438, upper bound: 17.9624438

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9623897, upper bound: 17.9612123
time: 0.50 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9615450
time: 0.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.24 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 3, lower bound: -17.9623897, upper bound: 17.9612123
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9615450

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.7510505, 7.4405704, -9.3307257, 7.8515010
1: -5.1917362, 7.8175325, -7.5991325, 11.2530985, -16.4448357, 15.4166651
2: -3.2490828, 7.0843124, -4.7880874, 10.3202887, -13.5693712, 11.8724003
3: -5.8050518, 8.6047878, -8.5250254, 12.5358915, -18.3409424, 17.1298141
4: -3.7534659, 8.5734529, -5.5422077, 12.5079746, -16.2614403, 14.1156607

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
time: 0.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
time: 0.66 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.4610372, 6.5926552, -2.7510505, 7.4405704, -9.9016075, 9.3437052
1: -6.7975974, 10.0043488, -7.5991325, 11.2530985, -18.0506954, 17.6034813
2: -4.2729445, 9.1532860, -4.7880874, 10.3202887, -14.5932331, 13.9413729
3: -7.6107635, 11.1077843, -8.5250254, 12.5358915, -20.1466522, 19.6328087
4: -4.9338398, 11.0912857, -5.5422077, 12.5079746, -17.4418144, 16.6334934

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
time: 0.52 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.20 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.8901551, 5.1004510, -6.9906063, 6.9906063
1: -5.1917362, 7.8175325, -5.1917362, 7.8175325, -13.0092678, 13.0092678
2: -3.2490828, 7.0843124, -3.2490828, 7.0843124, -10.3333950, 10.3333950
3: -5.8050518, 8.6047878, -5.8050518, 8.6047878, -14.4098396, 14.4098396
4: -3.7534659, 8.5734529, -3.7534659, 8.5734529, -12.3269186, 12.3269186

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620398, upper bound: 17.9611604
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
time: 0.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.4610372, 6.5926552, -8.4828100, 7.5614882
1: -5.1917362, 7.8175325, -6.7975974, 10.0043488, -15.1960831, 14.6151295
2: -3.2490828, 7.0843124, -4.2729445, 9.1532860, -12.4023685, 11.3572569
3: -5.8050518, 8.6047878, -7.6107635, 11.1077843, -16.9128361, 16.2155476
4: -3.7534659, 8.5734529, -4.9338398, 11.0912857, -14.8447514, 13.5072927

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9620398, upper bound: 17.9611604
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
time: 0.87 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.4610372, 6.5926552, -1.8901551, 5.1004510, -7.5614882, 8.4828100
1: -6.7975974, 10.0043488, -5.1917362, 7.8175325, -14.6151295, 15.1960831
2: -4.2729445, 9.1532860, -3.2490828, 7.0843124, -11.3572569, 12.4023685
3: -7.6107635, 11.1077843, -5.8050518, 8.6047878, -16.2155476, 16.9128361
4: -4.9338398, 11.0912857, -3.7534659, 8.5734529, -13.5072927, 14.8447514

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450
time: 0.57 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.4610372, 6.5926552, -2.4610372, 6.5926552, -9.0536909, 9.0536909
1: -6.7975974, 10.0043488, -6.7975974, 10.0043488, -16.8019466, 16.8019466
2: -4.2729445, 9.1532860, -4.2729445, 9.1532860, -13.4262304, 13.4262304
3: -7.6107635, 11.1077843, -7.6107635, 11.1077843, -18.7185440, 18.7185440
4: -4.9338398, 11.0912857, -4.9338398, 11.0912857, -16.0251255, 16.0251255

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450
time: 0.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.06 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9620398, upper bound: 17.9611604
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9620398, upper bound: 17.9611604
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -1.8901551, 5.1004510, -6.7420483, 6.3492041
1: -4.4953952, 6.8639817, -5.1917362, 7.8175325, -12.3129272, 12.0557156
2: -2.8065214, 6.1872668, -3.2490828, 7.0843124, -9.8908329, 9.4363470
3: -5.0186200, 7.5167313, -5.8050518, 8.6047878, -13.6234074, 13.3217831
4: -3.2377799, 7.4721994, -3.7534659, 8.5734529, -11.8112307, 11.2256651

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618155
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618155
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -1.8901551, 5.1004510, -6.8818808, 6.6855888
1: -4.8894196, 7.3645873, -5.1917362, 7.8175325, -12.7069521, 12.5563211
2: -3.0549450, 6.6544957, -3.2490828, 7.0843124, -10.1392555, 9.9035788
3: -5.4630013, 8.0912886, -5.8050518, 8.6047878, -14.0677872, 13.8963404
4: -3.5216219, 8.0479536, -3.7534659, 8.5734529, -12.0950747, 11.8014193

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618502
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618502
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -2.4610372, 6.5926552, -8.2342529, 6.9200859
1: -4.4953952, 6.8639817, -6.7975974, 10.0043488, -14.4997435, 13.6615791
2: -2.8065214, 6.1872668, -4.2729445, 9.1532860, -11.9598064, 10.4602098
3: -5.0186200, 7.5167313, -7.6107635, 11.1077843, -16.1264038, 15.1274920
4: -3.2377799, 7.4721994, -4.9338398, 11.0912857, -14.3290634, 12.4060383

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522251
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9611604
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -2.4610372, 6.5926552, -8.3740854, 7.2564707
1: -4.8894196, 7.3645873, -6.7975974, 10.0043488, -14.8937683, 14.1621847
2: -3.0549450, 6.6544957, -4.2729445, 9.1532860, -12.2082310, 10.9274406
3: -5.4630013, 8.0912886, -7.6107635, 11.1077843, -16.5707836, 15.7020502
4: -3.5216219, 8.0479536, -4.9338398, 11.0912857, -14.6129065, 12.9817924

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522597
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9611951
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.0445442, 5.4841022, -1.8901551, 5.1004510, -7.1449952, 7.3742571
1: -5.6337681, 8.3748035, -5.1917362, 7.8175325, -13.4512997, 13.5665379
2: -3.5288885, 7.6191778, -3.2490828, 7.0843124, -10.6132011, 10.8682585
3: -6.3012056, 9.2464848, -5.8050518, 8.6047878, -14.9059935, 15.0515366
4: -4.0707402, 9.2292709, -3.7534659, 8.5734529, -12.6441927, 12.9827366

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9522251, upper bound: 17.9616311
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9522251, upper bound: 17.9616311
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.3734298, 6.3504786, -1.8901551, 5.1004510, -7.4738808, 8.2406340
1: -6.5536866, 9.6473646, -5.1917362, 7.8175325, -14.3712196, 14.8390989
2: -4.1152163, 8.8181734, -3.2490828, 7.0843124, -11.1995277, 12.0672550
3: -7.3351488, 10.6968727, -5.8050518, 8.6047878, -15.9399347, 16.5019245
4: -4.7487473, 10.6886234, -3.7534659, 8.5734529, -13.3221998, 14.4420891

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611604, upper bound: 17.9622000
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611604, upper bound: 17.9622000
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.0445442, 5.4841022, -2.4610372, 6.5926552, -8.6371975, 7.9451389
1: -5.6337681, 8.3748035, -6.7975974, 10.0043488, -15.6381149, 15.1724014
2: -3.5288885, 7.6191778, -4.2729445, 9.1532860, -12.6821747, 11.8921213
3: -6.3012056, 9.2464848, -7.6107635, 11.1077843, -17.4089890, 16.8572407
4: -4.0707402, 9.2292709, -4.9338398, 11.0912857, -15.1620245, 14.1631107

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9520406, upper bound: 17.9520406
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9520406, upper bound: 17.9609760
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.3734298, 6.3504786, -2.4610372, 6.5926552, -8.9660835, 8.8115158
1: -6.5536866, 9.6473646, -6.7975974, 10.0043488, -16.5580349, 16.4449615
2: -4.1152163, 8.8181734, -4.2729445, 9.1532860, -13.2685013, 13.0911169
3: -7.3351488, 10.6968727, -7.6107635, 11.1077843, -18.4429321, 18.3076363
4: -4.7487473, 10.6886234, -4.9338398, 11.0912857, -15.8400316, 15.6224632

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609760, upper bound: 17.9526096
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609760, upper bound: 17.9615450
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.38 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618155
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618155
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618502
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618502
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522251
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9611604
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522597
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9611951
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9522251, upper bound: 17.9616311
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9522251, upper bound: 17.9616311
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9611604, upper bound: 17.9622000
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9611604, upper bound: 17.9622000
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9520406, upper bound: 17.9520406
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9520406, upper bound: 17.9609760
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9609760, upper bound: 17.9526096
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 3, lower bound: -17.9609760, upper bound: 17.9615450

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -1.6415975, 4.4590487, -6.1006460, 6.1006460
1: -4.4953952, 6.8639817, -4.4953952, 6.8639817, -11.3593760, 11.3593750
2: -2.8065214, 6.1872668, -2.8065214, 6.1872668, -8.9937868, 8.9937859
3: -5.0186200, 7.5167313, -5.0186200, 7.5167313, -12.5353508, 12.5353508
4: -3.2377799, 7.4721994, -3.2377799, 7.4721994, -10.7099771, 10.7099771

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9577431, upper bound: 17.9613728
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9619958, upper bound: 17.9618027
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -1.7814304, 4.7954335, -6.4370308, 6.2404790
1: -4.4953952, 6.8639817, -4.8894196, 7.3645873, -11.8599815, 11.7534008
2: -2.8065214, 6.1872668, -3.0549450, 6.6544957, -9.4610157, 9.2422085
3: -5.0186200, 7.5167313, -5.4630013, 8.0912886, -13.1099081, 12.9797287
4: -3.2377799, 7.4721994, -3.5216219, 8.0479536, -11.2857304, 10.9938211

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9577431, upper bound: 17.9613728
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9619958, upper bound: 17.9618027
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -1.6415975, 4.4590487, -6.2404790, 6.4370308
1: -4.8894196, 7.3645873, -4.4953952, 6.8639817, -11.7534008, 11.8599806
2: -3.0549450, 6.6544957, -2.8065214, 6.1872668, -9.2422085, 9.4610167
3: -5.4630013, 8.0912886, -5.0186200, 7.5167313, -12.9797287, 13.1099091
4: -3.5216219, 8.0479536, -3.2377799, 7.4721994, -10.9938202, 11.2857304

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9615822
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9616486
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -1.7814304, 4.7954335, -6.5768633, 6.5768633
1: -4.8894196, 7.3645873, -4.8894196, 7.3645873, -12.2540073, 12.2540073
2: -3.0549450, 6.6544957, -3.0549450, 6.6544957, -9.7094393, 9.7094393
3: -5.4630013, 8.0912886, -5.4630013, 8.0912886, -13.5542870, 13.5542870
4: -3.5216219, 8.0479536, -3.5216219, 8.0479536, -11.5695744, 11.5695744

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9615822
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9616486
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -2.0445442, 5.4841022, -7.1257000, 6.5035930
1: -4.4953952, 6.8639817, -5.6337681, 8.3748035, -12.8701992, 12.4977465
2: -2.8065214, 6.1872668, -3.5288885, 7.6191778, -10.4256973, 9.7161531
3: -5.0186200, 7.5167313, -6.3012056, 9.2464848, -14.2651033, 13.8179350
4: -3.2377799, 7.4721994, -4.0707402, 9.2292709, -12.4670496, 11.5429382

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9575585, upper bound: 17.9517393
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618112, upper bound: 17.9521692
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -2.3734298, 6.3504786, -7.9920759, 6.8324785
1: -4.4953952, 6.8639817, -6.5536866, 9.6473646, -14.1427593, 13.4176683
2: -2.8065214, 6.1872668, -4.1152163, 8.8181734, -11.6246929, 10.3024807
3: -5.0186200, 7.5167313, -7.3351488, 10.6968727, -15.7154922, 14.8518772
4: -3.2377799, 7.4721994, -4.7487473, 10.6886234, -13.9264030, 12.2209454

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9575585, upper bound: 17.9607127
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618112, upper bound: 17.9611426
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -2.0445442, 5.4841022, -7.2655320, 6.8399777
1: -4.8894196, 7.3645873, -5.6337681, 8.3748035, -13.2642231, 12.9983530
2: -3.0549450, 6.6544957, -3.5288885, 7.6191778, -10.6741199, 10.1833820
3: -5.4630013, 8.0912886, -6.3012056, 9.2464848, -14.7094812, 14.3924932
4: -3.5216219, 8.0479536, -4.0707402, 9.2292709, -12.7508926, 12.1186914

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9581251, upper bound: 17.9515161
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9515825
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -2.3734298, 6.3504786, -8.1319094, 7.1688633
1: -4.8894196, 7.3645873, -6.5536866, 9.6473646, -14.5367842, 13.9182739
2: -3.0549450, 6.6544957, -4.1152163, 8.8181734, -11.8731155, 10.7697096
3: -5.4630013, 8.0912886, -7.3351488, 10.6968727, -16.1598740, 15.4264345
4: -3.5216219, 8.0479536, -4.7487473, 10.6886234, -14.2102451, 12.7966986

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9581251, upper bound: 17.9604952
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9606334
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.0445442, 5.4841022, -1.6415975, 4.4590487, -6.5035930, 7.1257000
1: -5.6337681, 8.3748035, -4.4953952, 6.8639817, -12.4977484, 12.8701992
2: -3.5288885, 7.6191778, -2.8065214, 6.1872668, -9.7161531, 10.4256973
3: -6.3012056, 9.2464848, -5.0186200, 7.5167313, -13.8179350, 14.2651043
4: -4.0707402, 9.2292709, -3.2377799, 7.4721994, -11.5429392, 12.4670506

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9614295
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.0445442, 5.4841022, -1.7814304, 4.7954335, -6.8399777, 7.2655320
1: -5.6337681, 8.3748035, -4.8894196, 7.3645873, -12.9983521, 13.2642231
2: -3.5288885, 7.6191778, -3.0549450, 6.6544957, -10.1833830, 10.6741199
3: -6.3012056, 9.2464848, -5.4630013, 8.0912886, -14.3924932, 14.7094812
4: -4.0707402, 9.2292709, -3.5216219, 8.0479536, -12.1186914, 12.7508926

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515478, upper bound: 17.9614295
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.3734298, 6.3504786, -1.6415975, 4.4590487, -6.8324785, 7.9920759
1: -6.5536866, 9.6473646, -4.4953952, 6.8639817, -13.4176683, 14.1427593
2: -4.1152163, 8.8181734, -2.8065214, 6.1872668, -10.3024807, 11.6246939
3: -7.3351488, 10.6968727, -5.0186200, 7.5167313, -14.8518772, 15.7154922
4: -4.7487473, 10.6886234, -3.2377799, 7.4721994, -12.2209454, 13.9264030

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576424, upper bound: 17.9618297
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619985
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.3734298, 6.3504786, -1.7814304, 4.7954335, -7.1688633, 8.1319094
1: -6.5536866, 9.6473646, -4.8894196, 7.3645873, -13.9182739, 14.5367842
2: -4.1152163, 8.8181734, -3.0549450, 6.6544957, -10.7697115, 11.8731165
3: -7.3351488, 10.6968727, -5.4630013, 8.0912886, -15.4264345, 16.1598721
4: -4.7487473, 10.6886234, -3.5216219, 8.0479536, -12.7966986, 14.2102451

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576424, upper bound: 17.9618297
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619985
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.0445442, 5.4841022, -2.3734298, 6.3504786, -8.3950233, 7.8575320
1: -5.6337681, 8.3748035, -6.5536866, 9.6473646, -15.2811317, 14.9284897
2: -3.5288885, 7.6191778, -4.1152163, 8.8181734, -12.3470612, 11.7343922
3: -6.3012056, 9.2464848, -7.3351488, 10.6968727, -16.9980774, 16.5816288
4: -4.0707402, 9.2292709, -4.7487473, 10.6886234, -14.7593632, 13.9780178

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9478981, upper bound: 17.9607312
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9513634, upper bound: 17.9608225
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.3734298, 6.3504786, -2.0445442, 5.4841022, -7.8575320, 8.3950224
1: -6.5536866, 9.6473646, -5.6337681, 8.3748035, -14.9284897, 15.2811308
2: -4.1152163, 8.8181734, -3.5288885, 7.6191778, -11.7343922, 12.3470602
3: -7.3351488, 10.6968727, -6.3012056, 9.2464848, -16.5816288, 16.9980774
4: -4.7487473, 10.6886234, -4.0707402, 9.2292709, -13.9780178, 14.7593632

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9574580, upper bound: 17.9517636
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9519324
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.3734298, 6.3504786, -2.3734298, 6.3504786, -8.7239084, 8.7239084
1: -6.5536866, 9.6473646, -6.5536866, 9.6473646, -16.2010517, 16.2010498
2: -4.1152163, 8.8181734, -4.1152163, 8.8181734, -12.9333878, 12.9333858
3: -7.3351488, 10.6968727, -7.3351488, 10.6968727, -18.0320206, 18.0320206
4: -4.7487473, 10.6886234, -4.7487473, 10.6886234, -15.4373703, 15.4373703

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9574580, upper bound: 17.9612227
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9613915
time: 0.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.26 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9577431, upper bound: 17.9613728
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9619958, upper bound: 17.9618027
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9577431, upper bound: 17.9613728
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9619958, upper bound: 17.9618027
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9615822
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9616486
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9615822
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9616486
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9575585, upper bound: 17.9517393
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9618112, upper bound: 17.9521692
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9575585, upper bound: 17.9607127
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9618112, upper bound: 17.9611426
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9581251, upper bound: 17.9515161
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9515825
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9581251, upper bound: 17.9604952
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9606334
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9614295
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9515478, upper bound: 17.9614295
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9576424, upper bound: 17.9618297
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619985
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9576424, upper bound: 17.9618297
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619985
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9478981, upper bound: 17.9607312
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9513634, upper bound: 17.9608225
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9574580, upper bound: 17.9517636
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9519324
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9574580, upper bound: 17.9612227
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9613915

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.4799268, 4.0190759, -1.6415975, 4.4590487, -5.9389753, 5.6606731
1: -4.0262055, 6.2050543, -4.4953952, 6.8639817, -10.8901854, 10.7004490
2: -2.5223098, 5.5672717, -2.8065214, 6.1872668, -8.7095766, 8.3737917
3: -4.4950457, 6.7965755, -5.0186200, 7.5167313, -12.0117741, 11.8151951
4: -2.9154911, 6.7193985, -3.2377799, 7.4721994, -10.3876905, 9.9571772

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9573132, upper bound: 17.9573132
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9573132, upper bound: 17.9573132
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -1.6415975, 4.4590487, -6.0609984, 5.9878187
1: -4.3864923, 6.6942215, -4.4953952, 6.8639817, -11.2504721, 11.1896162
2: -2.7335219, 6.0273328, -2.8065214, 6.1872668, -8.9207878, 8.8338528
3: -4.8954883, 7.3203115, -5.0186200, 7.5167313, -12.4122171, 12.3389320
4: -3.1496921, 7.2866049, -3.2377799, 7.4721994, -10.6218910, 10.5243826

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615659, upper bound: 17.9577431
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615659, upper bound: 17.9619958
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.4799268, 4.0190759, -1.7814304, 4.7954335, -6.2753601, 5.8005052
1: -4.0262055, 6.2050543, -4.8894196, 7.3645873, -11.3907909, 11.0944738
2: -2.5223098, 5.5672717, -3.0549450, 6.6544957, -9.1768055, 8.6222153
3: -4.4950457, 6.7965755, -5.4630013, 8.0912886, -12.5863314, 12.2595739
4: -2.9154911, 6.7193985, -3.5216219, 8.0479536, -10.9634428, 10.2410202

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9573935, upper bound: 17.9573132
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9573935, upper bound: 17.9573132
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -1.7814304, 4.7954335, -6.3973832, 6.1276512
1: -4.3864923, 6.6942215, -4.8894196, 7.3645873, -11.7510757, 11.5836411
2: -2.7335219, 6.0273328, -3.0549450, 6.6544957, -9.3880177, 9.0822754
3: -4.8954883, 7.3203115, -5.4630013, 8.0912886, -12.9867754, 12.7833118
4: -3.1496921, 7.2866049, -3.5216219, 8.0479536, -11.1976442, 10.8082266

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615733, upper bound: 17.9577518
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617550, upper bound: 17.9616012
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.6045514, 4.3435383, -1.6415975, 4.4590487, -6.0636001, 5.9851360
1: -4.3935027, 6.6973906, -4.4953952, 6.8639817, -11.2574835, 11.1927853
2: -2.7491412, 6.0295582, -2.8065214, 6.1872668, -8.9364071, 8.8360786
3: -4.9114966, 7.3468657, -5.0186200, 7.5167313, -12.4282236, 12.3654861
4: -3.1771119, 7.2852683, -3.2377799, 7.4721994, -10.6493101, 10.5230465

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.6415975, 4.4590487, -6.2231841, 6.4159184
1: -4.8395271, 7.3438787, -4.4953952, 6.8639817, -11.7035084, 11.8392735
2: -3.0361805, 6.6364141, -2.8065214, 6.1872668, -9.2234440, 9.4429350
3: -5.4143553, 8.0728092, -5.0186200, 7.5167313, -12.9310837, 13.0914288
4: -3.5127008, 8.0279589, -3.2377799, 7.4721994, -10.9848986, 11.2657356

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9586849
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9617677
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6045514, 4.3435383, -1.7814304, 4.7954335, -6.3999848, 6.1249676
1: -4.3935027, 6.6973906, -4.8894196, 7.3645873, -11.7580891, 11.5868101
2: -2.7491412, 6.0295582, -3.0549450, 6.6544957, -9.4036369, 9.0845013
3: -4.9114966, 7.3468657, -5.4630013, 8.0912886, -13.0027809, 12.8098660
4: -3.1771119, 7.2852683, -3.5216219, 8.0479536, -11.2250643, 10.8068905

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9582777
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9615822
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.7814304, 4.7954335, -6.5595684, 6.5557508
1: -4.8395271, 7.3438787, -4.8894196, 7.3645873, -12.2041130, 12.2332983
2: -3.0361805, 6.6364141, -3.0549450, 6.6544957, -9.6906757, 9.6913576
3: -5.4143553, 8.0728092, -5.4630013, 8.0912886, -13.5056410, 13.5358086
4: -3.5127008, 8.0279589, -3.5216219, 8.0479536, -11.5606527, 11.5495796

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9583441
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9616485
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -2.0445442, 5.4841022, -7.0860515, 6.3907657
1: -4.3864923, 6.6942215, -5.6337681, 8.3748035, -12.7612934, 12.3279867
2: -2.7335219, 6.0273328, -3.5288885, 7.6191778, -10.3526993, 9.5562201
3: -4.8954883, 7.3203115, -6.3012056, 9.2464848, -14.1419716, 13.6215172
4: -3.1496921, 7.2866049, -4.0707402, 9.2292709, -12.3789635, 11.3573437

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613250, upper bound: 17.9471665
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615361, upper bound: 17.9514922
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.4799268, 4.0190759, -2.3734298, 6.3504786, -7.8304052, 6.3925052
1: -4.0262055, 6.2050543, -6.5536866, 9.6473646, -13.6735706, 12.7587414
2: -2.5223098, 5.5672717, -4.1152163, 8.8181734, -11.3404827, 9.6824875
3: -4.4950457, 6.7965755, -7.3351488, 10.6968727, -15.1919184, 14.1317234
4: -2.9154911, 6.7193985, -4.7487473, 10.6886234, -13.6041145, 11.4681454

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9577610, upper bound: 17.9569523
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9577610, upper bound: 17.9607127
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -2.3734298, 6.3504786, -7.9524274, 6.7196512
1: -4.3864923, 6.6942215, -6.5536866, 9.6473646, -14.0338535, 13.2479076
2: -2.7335219, 6.0273328, -4.1152163, 8.8181734, -11.5516930, 10.1425476
3: -4.8954883, 7.3203115, -7.3351488, 10.6968727, -15.5923615, 14.6554585
4: -3.1496921, 7.2866049, -4.7487473, 10.6886234, -13.8383160, 12.0353518

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618030, upper bound: 17.9568904
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9621011, upper bound: 17.9609893
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -2.0445442, 5.4841022, -7.2482376, 6.8188648
1: -4.8395271, 7.3438787, -5.6337681, 8.3748035, -13.2143307, 12.9776449
2: -3.0361805, 6.6364141, -3.5288885, 7.6191778, -10.6553555, 10.1653023
3: -5.4143553, 8.0728092, -6.3012056, 9.2464848, -14.6608372, 14.3740149
4: -3.5127008, 8.0279589, -4.0707402, 9.2292709, -12.7419710, 12.0986977

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9481172
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9515825
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -2.3734298, 6.3504786, -8.1146135, 7.1477504
1: -4.8395271, 7.3438787, -6.5536866, 9.6473646, -14.4868917, 13.8975658
2: -3.0361805, 6.6364141, -4.1152163, 8.8181734, -11.8543510, 10.7516298
3: -5.4143553, 8.0728092, -7.3351488, 10.6968727, -16.1112289, 15.4079580
4: -3.5127008, 8.0279589, -4.7487473, 10.6886234, -14.2013245, 12.7767048

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9575141
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9575142
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.6415975, 4.4590487, -6.3128986, 6.6254978
1: -5.0963554, 7.6366177, -4.4953952, 6.8639817, -11.9603348, 12.1320124
2: -3.2001376, 6.9270759, -2.8065214, 6.1872668, -9.3874044, 9.7335968
3: -5.7014980, 8.4203701, -5.0186200, 7.5167313, -13.2182264, 13.4389896
4: -3.6966069, 8.3848782, -3.2377799, 7.4721994, -11.1688061, 11.6226559

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9583745
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9583745
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.6415975, 4.4590487, -6.4848871, 7.0974216
1: -5.5796051, 8.3446941, -4.4953952, 6.8639817, -12.4435863, 12.8400888
2: -3.5107706, 7.5853653, -2.8065214, 6.1872668, -9.6980362, 10.3918858
3: -6.2471113, 9.2124949, -5.0186200, 7.5167313, -13.7638407, 14.2311153
4: -4.0569630, 9.1839800, -3.2377799, 7.4721994, -11.5291624, 12.4217577

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9584659
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9615486
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.7814304, 4.7954335, -6.6492834, 6.7653308
1: -5.0963554, 7.6366177, -4.8894196, 7.3645873, -12.4609413, 12.5260372
2: -3.2001376, 6.9270759, -3.0549450, 6.6544957, -9.8546333, 9.9820194
3: -5.7014980, 8.4203701, -5.4630013, 8.0912886, -13.7927847, 13.8833694
4: -3.6966069, 8.3848782, -3.5216219, 8.0479536, -11.7445602, 11.9064989

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9580337
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9613381
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.7814304, 4.7954335, -6.8212719, 7.2372541
1: -5.5796051, 8.3446941, -4.8894196, 7.3645873, -12.9441910, 13.2341137
2: -3.5107706, 7.5853653, -3.0549450, 6.6544957, -10.1652660, 10.6403093
3: -6.2471113, 9.2124949, -5.4630013, 8.0912886, -14.3383989, 14.6754932
4: -4.0569630, 9.1839800, -3.5216219, 8.0479536, -12.1049166, 12.7055998

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9581251
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9614295
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.1817567, 5.8508301, -1.6415975, 4.4590487, -6.6408052, 7.4924278
1: -6.0132799, 8.9072704, -4.4953952, 6.8639817, -12.8772621, 13.4026651
2: -3.7834954, 8.1247444, -2.8065214, 6.1872668, -9.9707623, 10.9312658
3: -6.7310286, 9.8678417, -5.0186200, 7.5167313, -14.2477579, 14.8864613
4: -4.3695273, 9.8453188, -3.2377799, 7.4721994, -11.8417263, 13.0830965

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9588660
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9588660
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.6415975, 4.4590487, -6.8116817, 7.9558506
1: -6.4942160, 9.6076832, -4.4953952, 6.8639817, -13.3581963, 14.1030779
2: -4.0940924, 8.7726021, -2.8065214, 6.1872668, -10.2813578, 11.5791225
3: -7.2752800, 10.6551113, -5.0186200, 7.5167313, -14.7920084, 15.6737309
4: -4.7291746, 10.6296968, -3.2377799, 7.4721994, -12.2013741, 13.8674755

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9590348
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9621175
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.1817567, 5.8508301, -1.7814304, 4.7954335, -6.9771900, 7.6322598
1: -6.0132799, 8.9072704, -4.8894196, 7.3645873, -13.3778667, 13.7966900
2: -3.7834954, 8.1247444, -3.0549450, 6.6544957, -10.4379911, 11.1796894
3: -6.7310286, 9.8678417, -5.4630013, 8.0912886, -14.8223152, 15.3308430
4: -4.3695273, 9.8453188, -3.5216219, 8.0479536, -12.4174805, 13.3669405

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9585253
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9585253
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7814304, 4.7954335, -7.1480665, 8.0956831
1: -6.4942160, 9.6076832, -4.8894196, 7.3645873, -13.8588028, 14.4971027
2: -4.0940924, 8.7726021, -3.0549450, 6.6544957, -10.7485886, 11.8275461
3: -7.2752800, 10.6551113, -5.4630013, 8.0912886, -15.3665648, 16.1181107
4: -4.7291746, 10.6296968, -3.5216219, 8.0479536, -12.7771263, 14.1513186

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9586940
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9619984
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -2.3734298, 6.3504786, -8.2043285, 7.3573303
1: -5.0963554, 7.6366177, -6.5536866, 9.6473646, -14.7437201, 14.1903038
2: -3.2001376, 6.9270759, -4.1152163, 8.8181734, -12.0183105, 11.0422907
3: -5.7014980, 8.4203701, -7.3351488, 10.6968727, -16.3983707, 15.7555189
4: -3.6966069, 8.3848782, -4.7487473, 10.6886234, -14.3852301, 13.1336241

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -2.3734298, 6.3504786, -8.3763170, 7.8292542
1: -5.5796051, 8.3446941, -6.5536866, 9.6473646, -15.2269688, 14.8983803
2: -3.5107706, 7.5853653, -4.1152163, 8.8181734, -12.3289433, 11.7005796
3: -6.2471113, 9.2124949, -7.3351488, 10.6968727, -16.9439850, 16.5476418
4: -4.0569630, 9.1839800, -4.7487473, 10.6886234, -14.7455864, 13.9327259

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9574580
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9574580
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -2.0445442, 5.4841022, -7.8367352, 8.3587952
1: -6.4942160, 9.6076832, -5.6337681, 8.3748035, -14.8690195, 15.2414503
2: -4.0940924, 8.7726021, -3.5288885, 7.6191778, -11.7132692, 12.3014898
3: -7.2752800, 10.6551113, -6.3012056, 9.2464848, -16.5217590, 16.9563179
4: -4.7291746, 10.6296968, -4.0707402, 9.2292709, -13.9584455, 14.7004366

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9484671
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9519324
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.1817567, 5.8508301, -2.3734298, 6.3504786, -8.5322351, 8.2242603
1: -6.0132799, 8.9072704, -6.5536866, 9.6473646, -15.6606445, 15.4609566
2: -3.7834954, 8.1247444, -4.1152163, 8.8181734, -12.6016693, 12.2399607
3: -6.7310286, 9.8678417, -7.3351488, 10.6968727, -17.4279022, 17.2029915
4: -4.3695273, 9.8453188, -4.7487473, 10.6886234, -15.0581512, 14.5940657

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -2.3734298, 6.3504786, -8.7031116, 8.6876812
1: -6.4942160, 9.6076832, -6.5536866, 9.6473646, -16.1415806, 16.1613693
2: -4.0940924, 8.7726021, -4.1152163, 8.8181734, -12.9122658, 12.8878164
3: -7.2752800, 10.6551113, -7.3351488, 10.6968727, -17.9721527, 17.9902611
4: -4.7291746, 10.6296968, -4.7487473, 10.6886234, -15.4177980, 15.3784447

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9580269
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9613915
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.26 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9573132, upper bound: 17.9573132
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9573132, upper bound: 17.9573132
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615659, upper bound: 17.9577431
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615659, upper bound: 17.9619958
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9573935, upper bound: 17.9573132
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9573935, upper bound: 17.9573132
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615733, upper bound: 17.9577518
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9617550, upper bound: 17.9616012
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
NS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9586849
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9617677
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9582777
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9615822
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9583441
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9616485
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9613250, upper bound: 17.9471665
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9615361, upper bound: 17.9514922
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9577610, upper bound: 17.9569523
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9577610, upper bound: 17.9607127
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9618030, upper bound: 17.9568904
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9621011, upper bound: 17.9609893
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9481172
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9515825
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9575141
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9575142
NS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9583745
NS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9583745
NS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9584659
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9615486
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9580337
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9613381
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9581251
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9614295
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9588660
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9588660
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9590348
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9621175
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9585253
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9585253
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9586940
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9619984
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9574580
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9574580
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9484671
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9519324
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9580269
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.26
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9613915

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -1.4799268, 4.0190759, -5.6210246, 5.8261480
1: -4.3864923, 6.6942215, -4.0262055, 6.2050543, -10.5915451, 10.7204266
2: -2.7335219, 6.0273328, -2.5223098, 5.5672717, -8.3007927, 8.5496426
3: -4.8954883, 7.3203115, -4.4950457, 6.7965755, -11.6920633, 11.8153553
4: -3.1496921, 7.2866049, -2.9154911, 6.7193985, -9.8690910, 10.2020960

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614282, upper bound: 17.9576038
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609138, upper bound: 17.9574641
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -1.6019502, 4.3462214, -5.9481707, 5.9481707
1: -4.3864923, 6.6942215, -4.3864923, 6.6942215, -11.0807123, 11.0807133
2: -2.7335219, 6.0273328, -2.7335219, 6.0273328, -8.7608547, 8.7608538
3: -4.8954883, 7.3203115, -4.8954883, 7.3203115, -12.2157993, 12.2157993
4: -3.1496921, 7.2866049, -3.1496921, 7.2866049, -10.4362965, 10.4362965

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614282, upper bound: 17.9576039
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609138, upper bound: 17.9609822
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -1.6045514, 4.3435383, -5.9454875, 5.9507728
1: -4.3864923, 6.6942215, -4.3935027, 6.6973906, -11.0838814, 11.0877247
2: -2.7335219, 6.0273328, -2.7491412, 6.0295582, -8.7630787, 8.7764740
3: -4.8954883, 7.3203115, -4.9114966, 7.3468657, -12.2423534, 12.2318048
4: -3.1496921, 7.2866049, -3.1771119, 7.2852683, -10.4349604, 10.4637165

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9579976, upper bound: 17.9575725
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9579976, upper bound: 17.9577518
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -1.7641354, 4.7743206, -6.3762708, 6.1103568
1: -4.3864923, 6.6942215, -4.8395271, 7.3438787, -11.7303696, 11.5337486
2: -2.7335219, 6.0273328, -3.0361805, 6.6364141, -9.3699360, 9.0635109
3: -4.8954883, 7.3203115, -5.4143553, 8.0728092, -12.9682980, 12.7346649
4: -3.1496921, 7.2866049, -3.5127008, 8.0279589, -11.1776505, 10.7993050

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9581793, upper bound: 17.9614220
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9581793, upper bound: 17.9616013
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.4669125, 4.0071440, -5.7712784, 6.2412329
1: -4.8395271, 7.3438787, -4.0064602, 6.2006745, -11.0402012, 11.3503389
2: -3.0361805, 6.6364141, -2.5055699, 5.5651913, -8.6013708, 9.1419840
3: -5.4143553, 8.0728092, -4.4741712, 6.7811985, -12.1955519, 12.5469799
4: -3.5127008, 8.0279589, -2.8958118, 6.7110949, -10.2237959, 10.9237700

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601868, upper bound: 17.9580875
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615472, upper bound: 17.9586849
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.6265631, 4.4408922, -6.2050276, 6.4008837
1: -4.8395271, 7.3438787, -4.4537382, 6.8536711, -11.6931982, 11.7976170
2: -3.0361805, 6.6364141, -2.7905805, 6.1733203, -9.2095003, 9.4269943
3: -5.4143553, 8.0728092, -4.9789329, 7.5049791, -12.9193335, 13.0517426
4: -3.5127008, 8.0279589, -3.2266197, 7.4491405, -10.9618397, 11.2545776

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613912, upper bound: 17.9582977
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9609762
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6045514, 4.3435383, -1.7641354, 4.7743206, -6.3788719, 6.1076732
1: -4.3935027, 6.6973906, -4.8395271, 7.3438787, -11.7373810, 11.5369177
2: -2.7491412, 6.0295582, -3.0361805, 6.6364141, -9.3855553, 9.0657358
3: -4.9114966, 7.3468657, -5.4143553, 8.0728092, -12.9843044, 12.7612209
4: -3.1771119, 7.2852683, -3.5127008, 8.0279589, -11.2050705, 10.7979679

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9563885, upper bound: 17.9573484
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9575725, upper bound: 17.9614540
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.6045514, 4.3435383, -6.1076732, 6.3788719
1: -4.8395271, 7.3438787, -4.3935027, 6.6973906, -11.5369177, 11.7373810
2: -3.0361805, 6.6364141, -2.7491412, 6.0295582, -9.0657358, 9.3855553
3: -5.4143553, 8.0728092, -4.9114966, 7.3468657, -12.7612209, 12.9843044
4: -3.5127008, 8.0279589, -3.1771119, 7.2852683, -10.7979689, 11.2050705

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9557599, upper bound: 17.9566387
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614219, upper bound: 17.9577862
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.7641354, 4.7743206, -6.5384560, 6.5384560
1: -4.8395271, 7.3438787, -4.8395271, 7.3438787, -12.1834059, 12.1834059
2: -3.0361805, 6.6364141, -3.0361805, 6.6364141, -9.6725922, 9.6725941
3: -5.4143553, 8.0728092, -5.4143553, 8.0728092, -13.4871645, 13.4871645
4: -3.5127008, 8.0279589, -3.5127008, 8.0279589, -11.5406580, 11.5406590

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9557599, upper bound: 17.9567388
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614220, upper bound: 17.9577862
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -1.8538499, 4.9839005, -6.5858498, 6.2000713
1: -4.3864923, 6.6942215, -5.0963554, 7.6366177, -12.0231075, 11.7905760
2: -2.7335219, 6.0273328, -3.2001376, 6.9270759, -9.6605978, 9.2274704
3: -4.8954883, 7.3203115, -5.7014980, 8.4203701, -13.3158588, 13.0218096
4: -3.1496921, 7.2866049, -3.6966069, 8.3848782, -11.5345697, 10.9832115

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9577493, upper bound: 17.9469872
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9577493, upper bound: 17.9471665
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -2.0258384, 5.4558244, -7.0577741, 6.3720598
1: -4.3864923, 6.6942215, -5.5796051, 8.3446941, -12.7311850, 12.2738256
2: -2.7335219, 6.0273328, -3.5107706, 7.5853653, -10.3188868, 9.5381031
3: -4.8954883, 7.3203115, -6.2471113, 9.2124949, -14.1079826, 13.5674229
4: -3.1496921, 7.2866049, -4.0569630, 9.1839800, -12.3336716, 11.3435678

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9579604, upper bound: 17.9513130
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9579604, upper bound: 17.9514923
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.4799268, 4.0190759, -2.3335648, 6.2360497, -7.7159767, 6.3526406
1: -4.0262055, 6.2050543, -6.4449759, 9.4759874, -13.5021925, 12.6500292
2: -2.5223098, 5.5672717, -4.0416408, 8.6598349, -11.1821442, 9.6089125
3: -4.4950457, 6.7965755, -7.2123494, 10.4980783, -14.9931240, 14.0089245
4: -2.9154911, 6.7193985, -4.6602254, 10.5010548, -13.4165449, 11.3796225

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9561677, upper bound: 17.9600149
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576259, upper bound: 17.9604250
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -2.1817567, 5.8508301, -7.4527798, 6.5279779
1: -4.3864923, 6.6942215, -6.0132799, 8.9072704, -13.2937613, 12.7075014
2: -2.7335219, 6.0273328, -3.7834954, 8.1247444, -10.8582668, 9.8108282
3: -4.8954883, 7.3203115, -6.7310286, 9.8678417, -14.7633305, 14.0513391
4: -3.1496921, 7.2866049, -4.3695273, 9.8453188, -12.9950104, 11.6561317

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582273, upper bound: 17.9567111
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582273, upper bound: 17.9568904
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.6019502, 4.3462214, -2.3526335, 6.3142529, -7.9162016, 6.6988544
1: -4.3864923, 6.6942215, -6.4942160, 9.6076832, -13.9941740, 13.1884375
2: -2.7335219, 6.0273328, -4.0940924, 8.7726021, -11.5061235, 10.1214237
3: -4.8954883, 7.3203115, -7.2752800, 10.6551113, -15.5506001, 14.5955896
4: -3.1496921, 7.2866049, -4.7291746, 10.6296968, -13.7793884, 12.0157795

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9585254, upper bound: 17.9608100
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9585254, upper bound: 17.9609893
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.8538499, 4.9839005, -6.7480354, 6.6281705
1: -4.8395271, 7.3438787, -5.0963554, 7.6366177, -12.4761448, 12.4402342
2: -3.0361805, 6.6364141, -3.2001376, 6.9270759, -9.9632540, 9.8365517
3: -5.4143553, 8.0728092, -5.7014980, 8.4203701, -13.8347254, 13.7743063
4: -3.5127008, 8.0279589, -3.6966069, 8.3848782, -11.8975773, 11.7245655

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9555116, upper bound: 17.9460534
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612056, upper bound: 17.9472009
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -2.0258384, 5.4558244, -7.2199593, 6.8001590
1: -4.8395271, 7.3438787, -5.5796051, 8.3446941, -13.1842213, 12.9234838
2: -3.0361805, 6.6364141, -3.5107706, 7.5853653, -10.6215439, 10.1471844
3: -5.4143553, 8.0728092, -6.2471113, 9.2124949, -14.6268492, 14.3199205
4: -3.5127008, 8.0279589, -4.0569630, 9.1839800, -12.6966782, 12.0849218

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9555116, upper bound: 17.9462657
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612056, upper bound: 17.9515267
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -2.1817567, 5.8508301, -7.6149650, 6.9560776
1: -4.8395271, 7.3438787, -6.0132799, 8.9072704, -13.7467966, 13.3571587
2: -3.0361805, 6.6364141, -3.7834954, 8.1247444, -11.1609249, 10.4199095
3: -5.4143553, 8.0728092, -6.7310286, 9.8678417, -15.2821970, 14.8038378
4: -3.5127008, 8.0279589, -4.3695273, 9.8453188, -13.3580189, 12.3974857

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9559460, upper bound: 17.9557773
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613881, upper bound: 17.9567014
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -2.3526335, 6.3142529, -8.0783873, 7.1269541
1: -4.8395271, 7.3438787, -6.4942160, 9.6076832, -14.4472103, 13.8380947
2: -3.0361805, 6.6364141, -4.0940924, 8.7726021, -11.8087797, 10.7305059
3: -5.4143553, 8.0728092, -7.2752800, 10.6551113, -16.0694656, 15.3480883
4: -3.5127008, 8.0279589, -4.7291746, 10.6296968, -14.1423969, 12.7571325

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9559460, upper bound: 17.9559030
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613881, upper bound: 17.9605291
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.6265631, 4.4408922, -6.4667306, 7.0823870
1: -5.5796051, 8.3446941, -4.4537382, 6.8536711, -12.4332762, 12.7984324
2: -3.5107706, 7.5853653, -2.7905805, 6.1733203, -9.6840906, 10.3759451
3: -6.2471113, 9.2124949, -4.9789329, 7.5049791, -13.7520905, 14.1914272
4: -4.0569630, 9.1839800, -3.2266197, 7.4491405, -11.5061035, 12.4105988

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9510596, upper bound: 17.9604944
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9510098, upper bound: 17.9584234
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9584659
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.7641354, 4.7743206, -6.6281705, 6.7480354
1: -5.0963554, 7.6366177, -4.8395271, 7.3438787, -12.4402342, 12.4761448
2: -3.2001376, 6.9270759, -3.0361805, 6.6364141, -9.8365517, 9.9632540
3: -5.7014980, 8.4203701, -5.4143553, 8.0728092, -13.7743063, 13.8347254
4: -3.6966069, 8.3848782, -3.5127008, 8.0279589, -11.7245655, 11.8975773

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9446201, upper bound: 17.9570049
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9470191, upper bound: 17.9612057
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.7641354, 4.7743206, -6.8001590, 7.2199593
1: -5.5796051, 8.3446941, -4.8395271, 7.3438787, -12.9234838, 13.1842213
2: -3.5107706, 7.5853653, -3.0361805, 6.6364141, -10.1471844, 10.6215439
3: -6.2471113, 9.2124949, -5.4143553, 8.0728092, -14.3199205, 14.6268492
4: -4.0569630, 9.1839800, -3.5127008, 8.0279589, -12.0849218, 12.6966782

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9510964, upper bound: 17.9604763
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9510447, upper bound: 17.9580826
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9610202
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.4669125, 4.0071440, -6.3597765, 7.7811642
1: -6.4942160, 9.6076832, -4.0064602, 6.2006745, -12.6948910, 13.6141434
2: -4.0940924, 8.7726021, -2.5055699, 5.5651913, -9.6592836, 11.2781715
3: -7.2752800, 10.6551113, -4.4741712, 6.7811985, -14.0564766, 15.1292820
4: -4.7291746, 10.6296968, -2.8958118, 6.7110949, -11.4402695, 13.5255089

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601887, upper bound: 17.9590348
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609402, upper bound: 17.9590293
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.6265631, 4.4408922, -6.7935257, 7.9408154
1: -6.4942160, 9.6076832, -4.4537382, 6.8536711, -13.3478870, 14.0614214
2: -4.0940924, 8.7726021, -2.7905805, 6.1733203, -10.2674122, 11.5631828
3: -7.2752800, 10.6551113, -4.9789329, 7.5049791, -14.7802591, 15.6340446
4: -4.7291746, 10.6296968, -3.2266197, 7.4491405, -12.1783142, 13.8563166

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601888, upper bound: 17.9621155
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9620625
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.6045514, 4.3435383, -6.6961713, 7.9188032
1: -6.4942160, 9.6076832, -4.3935027, 6.6973906, -13.1916065, 14.0011864
2: -4.0940924, 8.7726021, -2.7491412, 6.0295582, -10.1236486, 11.5217438
3: -7.2752800, 10.6551113, -4.9114966, 7.3468657, -14.6221447, 15.5666065
4: -4.7291746, 10.6296968, -3.1771119, 7.2852683, -12.0144424, 13.8068085

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9573333
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608419, upper bound: 17.9581323
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7641354, 4.7743206, -7.1269541, 8.0783882
1: -6.4942160, 9.6076832, -4.8395271, 7.3438787, -13.8380947, 14.4472103
2: -4.0940924, 8.7726021, -3.0361805, 6.6364141, -10.7305050, 11.8087797
3: -7.2752800, 10.6551113, -5.4143553, 8.0728092, -15.3480873, 16.0694656
4: -4.7291746, 10.6296968, -3.5127008, 8.0279589, -12.7571325, 14.1423969

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9573356
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608420, upper bound: 17.9581323
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.8538499, 4.9839005, -7.3365335, 8.1681023
1: -6.4942160, 9.6076832, -5.0963554, 7.6366177, -14.1308336, 14.7040386
2: -4.0940924, 8.7726021, -3.2001376, 6.9270759, -11.0211678, 11.9727402
3: -7.2752800, 10.6551113, -5.7014980, 8.4203701, -15.6956482, 16.3566093
4: -4.7291746, 10.6296968, -3.6966069, 8.3848782, -13.1140518, 14.3263035

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9554807, upper bound: 17.9467480
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605936, upper bound: 17.9475470
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -2.0258384, 5.4558244, -7.8084574, 8.3400917
1: -6.4942160, 9.6076832, -5.5796051, 8.3446941, -14.8389101, 15.1872873
2: -4.0940924, 8.7726021, -3.5107706, 7.5853653, -11.6794577, 12.2833729
3: -7.2752800, 10.6551113, -6.2471113, 9.2124949, -16.4877720, 16.9022217
4: -4.7291746, 10.6296968, -4.0569630, 9.1839800, -13.9131536, 14.6866598

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9554807, upper bound: 17.9468456
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605937, upper bound: 17.9518728
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -2.1817567, 5.8508301, -8.2034636, 8.4960089
1: -6.4942160, 9.6076832, -6.0132799, 8.9072704, -15.4014854, 15.6209631
2: -4.0940924, 8.7726021, -3.7834954, 8.1247444, -12.2188368, 12.5560970
3: -7.2752800, 10.6551113, -6.7310286, 9.8678417, -17.1431198, 17.3861389
4: -4.7291746, 10.6296968, -4.3695273, 9.8453188, -14.5744925, 14.9992237

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9564719
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608419, upper bound: 17.9572709
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -2.3526335, 6.3142529, -8.6668854, 8.6668854
1: -6.4942160, 9.6076832, -6.4942160, 9.6076832, -16.1018982, 16.1018982
2: -4.0940924, 8.7726021, -4.0940924, 8.7726021, -12.8666945, 12.8666945
3: -7.2752800, 10.6551113, -7.2752800, 10.6551113, -17.9303913, 17.9303913
4: -4.7291746, 10.6296968, -4.7291746, 10.6296968, -15.3588715, 15.3588715

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9564887
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608420, upper bound: 17.9572709
time: 0.53 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.18 seconds
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9614282, upper bound: 17.9576038
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9609138, upper bound: 17.9574641
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9614282, upper bound: 17.9576039
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9609138, upper bound: 17.9609822
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9579976, upper bound: 17.9575725
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9579976, upper bound: 17.9577518
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9581793, upper bound: 17.9614220
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9581793, upper bound: 17.9616013
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9601868, upper bound: 17.9580875
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9615472, upper bound: 17.9586849
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9613912, upper bound: 17.9582977
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9609762
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9563885, upper bound: 17.9573484
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9575725, upper bound: 17.9614540
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9557599, upper bound: 17.9566387
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9614219, upper bound: 17.9577862
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9557599, upper bound: 17.9567388
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9614220, upper bound: 17.9577862
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9577493, upper bound: 17.9469872
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9577493, upper bound: 17.9471665
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9579604, upper bound: 17.9513130
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9579604, upper bound: 17.9514923
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9561677, upper bound: 17.9600149
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9576259, upper bound: 17.9604250
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9582273, upper bound: 17.9567111
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9582273, upper bound: 17.9568904
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9585254, upper bound: 17.9608100
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9585254, upper bound: 17.9609893
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9555116, upper bound: 17.9460534
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9612056, upper bound: 17.9472009
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9555116, upper bound: 17.9462657
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9612056, upper bound: 17.9515267
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9559460, upper bound: 17.9557773
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9613881, upper bound: 17.9567014
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9559460, upper bound: 17.9559030
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9613881, upper bound: 17.9605291
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9510098, upper bound: 17.9584234
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9584659
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9446201, upper bound: 17.9570049
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9470191, upper bound: 17.9612057
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9510447, upper bound: 17.9580826
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9610202
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9601887, upper bound: 17.9590348
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9609402, upper bound: 17.9590293
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9601888, upper bound: 17.9621155
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9620625
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9573333
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9608419, upper bound: 17.9581323
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9573356
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9608420, upper bound: 17.9581323
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9554807, upper bound: 17.9467480
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9605936, upper bound: 17.9475470
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9554807, upper bound: 17.9468456
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9605937, upper bound: 17.9518728
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9564719
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9608419, upper bound: 17.9572709
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9557290, upper bound: 17.9564887
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 3, lower bound: -17.9608420, upper bound: 17.9572709

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.5188491, 4.1307187, -1.4799268, 4.0190759, -5.5379248, 5.6106453
1: -4.1579542, 6.3804936, -4.0262055, 6.2050543, -10.3630066, 10.4066973
2: -2.5872672, 5.7286654, -2.5223098, 5.5672717, -8.1545391, 8.2509747
3: -4.6399579, 6.9638162, -4.4950457, 6.7965755, -11.4365330, 11.4588614
4: -2.9800746, 6.9294696, -2.9154911, 6.7193985, -9.6994724, 9.8449602

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9560060
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9574641
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.5735397, 4.2779770, -1.4799268, 4.0190759, -5.5926147, 5.7579041
1: -4.3077784, 6.5949931, -4.0262055, 6.2050543, -10.5128307, 10.6211977
2: -2.6837308, 5.9328156, -2.5223098, 5.5672717, -8.2510023, 8.4551258
3: -4.8063774, 7.2089038, -4.4950457, 6.7965755, -11.6029530, 11.7039490
4: -3.0928497, 7.1728654, -2.9154911, 6.7193985, -9.8122482, 10.0883551

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9560060
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9574641
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.5188491, 4.1307187, -1.6019502, 4.3462214, -5.8650703, 5.7326684
1: -4.1579542, 6.3804936, -4.3864923, 6.6942215, -10.8521757, 10.7669830
2: -2.5872672, 5.7286654, -2.7335219, 6.0273328, -8.6145992, 8.4621859
3: -4.6399579, 6.9638162, -4.8954883, 7.3203115, -11.9602680, 11.8593044
4: -2.9800746, 6.9294696, -3.1496921, 7.2866049, -10.2666779, 10.0791607

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609821
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609821
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.5735397, 4.2779770, -1.6019502, 4.3462214, -5.9197607, 5.8799272
1: -4.3077784, 6.5949931, -4.3864923, 6.6942215, -11.0019989, 10.9814816
2: -2.6837308, 5.9328156, -2.7335219, 6.0273328, -8.7110634, 8.6663380
3: -4.8063774, 7.2089038, -4.8954883, 7.3203115, -12.1266890, 12.1043921
4: -3.0928497, 7.1728654, -3.1496921, 7.2866049, -10.3794546, 10.3225565

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609822
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609822
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.4266322, 3.8911672, -1.7641354, 4.7743206, -6.2009525, 5.6553025
1: -3.8957067, 6.0296102, -4.8395271, 7.3438787, -11.2395859, 10.8691368
2: -2.4311121, 5.4025393, -3.0361805, 6.6364141, -9.0675249, 8.4387178
3: -4.3494096, 6.5820875, -5.4143553, 8.0728092, -12.4222183, 11.9964428
4: -2.8053033, 6.5223231, -3.5127008, 8.0279589, -10.8332596, 10.0350237

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9570318, upper bound: 17.9557279
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9570318, upper bound: 17.9557279
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.5681590, 4.2715874, -1.7641354, 4.7743206, -6.3424797, 6.0357223
1: -4.2907004, 6.5990438, -4.8395271, 7.3438787, -11.6345787, 11.4385710
2: -2.6810930, 5.9358191, -3.0361805, 6.6364141, -9.3175058, 8.9719982
3: -4.7947497, 7.2176991, -5.4143553, 8.0728092, -12.8675585, 12.6320543
4: -3.0943124, 7.1687469, -3.5127008, 8.0279589, -11.1222706, 10.6814451

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9570318, upper bound: 17.9559072
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9570319, upper bound: 17.9614494
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7190464, 4.6567822, -1.4669125, 4.0071440, -5.7261896, 6.1236944
1: -4.7160978, 7.1701775, -4.0064602, 6.2006745, -10.9167728, 11.1766376
2: -2.9575763, 6.4752569, -2.5055699, 5.5651913, -8.5227680, 8.9808273
3: -5.2760663, 7.8768725, -4.4741712, 6.7811985, -12.0572634, 12.3510437
4: -3.4220204, 7.8321195, -2.8958118, 6.7110949, -10.1331158, 10.7279301

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610637, upper bound: 17.9584557
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9582616
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6740775, 4.5360212, -1.6265631, 4.4408922, -6.1149697, 6.1625843
1: -4.5909591, 6.9983420, -4.4537382, 6.8536711, -11.4446297, 11.4520798
2: -2.8773313, 6.3060794, -2.7905805, 6.1733203, -9.0506516, 9.0966597
3: -5.1356077, 7.6802788, -4.9789329, 7.5049791, -12.6405869, 12.6592121
4: -3.3281691, 7.6252813, -3.2266197, 7.4491405, -10.7773094, 10.8519011

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609762
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609762
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7382356, 4.7122388, -1.6265631, 4.4408922, -6.1791277, 6.3388019
1: -4.7675924, 7.2522669, -4.4537382, 6.8536711, -11.6212635, 11.7060051
2: -2.9908900, 6.5507212, -2.7905805, 6.1733203, -9.1642103, 9.3413019
3: -5.3331223, 7.9705186, -4.9789329, 7.5049791, -12.8381014, 12.9494514
4: -3.4612842, 7.9243174, -3.2266197, 7.4491405, -10.9104252, 11.1509361

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609761
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609761
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.5651021, 4.2343812, -1.7641354, 4.7743206, -6.3394227, 5.9985156
1: -4.2856231, 6.5295439, -4.8395271, 7.3438787, -11.6295013, 11.3690701
2: -2.6765900, 5.8743706, -3.0361805, 6.6364141, -9.3130026, 8.9105482
3: -4.7896037, 7.1508865, -5.4143553, 8.0728092, -12.8624105, 12.5652409
4: -3.0873482, 7.0983520, -3.5127008, 8.0279589, -11.1153069, 10.6110525

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9559578, upper bound: 17.9552620
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9559578, upper bound: 17.9614540
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.6045514, 4.3435383, -6.0446196, 6.1931934
1: -4.6642113, 7.0726247, -4.3935027, 6.6973906, -11.3615990, 11.4661264
2: -2.9182312, 6.3778706, -2.7491412, 6.0295582, -8.9477882, 9.1270123
3: -5.2167773, 7.7601538, -4.9114966, 7.3468657, -12.5636425, 12.6716480
4: -3.3682301, 7.7145586, -3.1771119, 7.2852683, -10.6534986, 10.8916702

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611313, upper bound: 17.9566023
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611313, upper bound: 17.9577862
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.7641354, 4.7743206, -6.4754019, 6.3527770
1: -4.6642113, 7.0726247, -4.8395271, 7.3438787, -12.0080891, 11.9121513
2: -2.9182312, 6.3778706, -3.0361805, 6.6364141, -9.5546446, 9.4140501
3: -5.2167773, 7.7601538, -5.4143553, 8.0728092, -13.2895861, 13.1745081
4: -3.3682301, 7.7145586, -3.5127008, 8.0279589, -11.3961887, 11.2272596

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9600314, upper bound: 17.9555465
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9600314, upper bound: 17.9615401
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.4266322, 3.8911672, -2.3526335, 6.3142529, -7.7408848, 6.2438006
1: -3.8957067, 6.0296102, -6.4942160, 9.6076832, -13.5033894, 12.5238266
2: -2.4311121, 5.4025393, -4.0940924, 8.7726021, -11.2037144, 9.4966288
3: -4.3494096, 6.5820875, -7.2752800, 10.6551113, -15.0045204, 13.8573675
4: -2.8053033, 6.5223231, -4.7291746, 10.6296968, -13.4349995, 11.2514963

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9556970
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9556970
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.5681590, 4.2715874, -2.3526335, 6.3142529, -7.8824110, 6.6242199
1: -4.2907004, 6.5990438, -6.4942160, 9.6076832, -13.8983841, 13.0932598
2: -2.6810930, 5.9358191, -4.0940924, 8.7726021, -11.4536943, 10.0299110
3: -4.7947497, 7.2176991, -7.2752800, 10.6551113, -15.4498615, 14.4929790
4: -3.0943124, 7.1687469, -4.7291746, 10.6296968, -13.7240095, 11.8979197

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9556970
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9567111
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.8538499, 4.9839005, -6.6849818, 6.4424920
1: -4.6642113, 7.0726247, -5.0963554, 7.6366177, -12.3008280, 12.1689787
2: -2.9182312, 6.3778706, -3.2001376, 6.9270759, -9.8453064, 9.5780087
3: -5.2167773, 7.7601538, -5.7014980, 8.4203701, -13.6371460, 13.4616508
4: -3.3682301, 7.7145586, -3.6966069, 8.3848782, -11.7531080, 11.4111652

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9448018
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9472009
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -2.0258384, 5.4558244, -7.1569057, 6.6144800
1: -4.6642113, 7.0726247, -5.5796051, 8.3446941, -13.0089035, 12.6522293
2: -2.9182312, 6.3778706, -3.5107706, 7.5853653, -10.5035954, 9.8886414
3: -5.2167773, 7.7601538, -6.2471113, 9.2124949, -14.4292707, 14.0072651
4: -3.3682301, 7.7145586, -4.0569630, 9.1839800, -12.5522099, 11.7715216

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609718, upper bound: 17.9510409
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613744, upper bound: 17.9510552
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614167, upper bound: 17.9515266
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -2.1817567, 5.8508301, -7.5519114, 6.7703986
1: -4.6642113, 7.0726247, -6.0132799, 8.9072704, -13.5714808, 13.0859051
2: -2.9182312, 6.3778706, -3.7834954, 8.1247444, -11.0429754, 10.1613655
3: -5.2167773, 7.7601538, -6.7310286, 9.8678417, -15.0846195, 14.4911814
4: -3.3682301, 7.7145586, -4.3695273, 9.8453188, -13.2135487, 12.0840855

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611245, upper bound: 17.9558464
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611245, upper bound: 17.9567014
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -2.3526335, 6.3142529, -8.0153341, 6.9412751
1: -4.6642113, 7.0726247, -6.4942160, 9.6076832, -14.2718935, 13.5668402
2: -2.9182312, 6.3778706, -4.0940924, 8.7726021, -11.6908331, 10.4719629
3: -5.2167773, 7.7601538, -7.2752800, 10.6551113, -15.8718872, 15.0354309
4: -3.3682301, 7.7145586, -4.7291746, 10.6296968, -13.9979267, 12.4437332

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9600393, upper bound: 17.9553769
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9600393, upper bound: 17.9605291
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.8139337, 4.8666945, -1.7641354, 4.7743206, -6.5882545, 6.6308298
1: -4.9869347, 7.4631629, -4.8395271, 7.3438787, -12.3308134, 12.3026905
2: -3.1260049, 6.7634478, -3.0361805, 6.6364141, -9.7624187, 9.7996283
3: -5.5779891, 8.2200403, -5.4143553, 8.0728092, -13.6507959, 13.6343956
4: -3.6069016, 8.1930723, -3.5127008, 8.0279589, -11.6348600, 11.7057714

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9460534, upper bound: 17.9555116
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9460534, upper bound: 17.9612056
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.0242922, 5.4517026, -1.7641354, 4.7743206, -6.7986126, 7.2158380
1: -5.5753055, 8.3385353, -4.8395271, 7.3438787, -12.9191837, 13.1780624
2: -3.5078702, 7.5796757, -3.0361805, 6.6364141, -10.1442823, 10.6158533
3: -6.2422271, 9.2054749, -5.4143553, 8.0728092, -14.3150368, 14.6198292
4: -4.0535598, 9.1772156, -3.5127008, 8.0279589, -12.0815182, 12.6899147

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9513839, upper bound: 17.9593479
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9513839, upper bound: 17.9610201
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.3027291, 6.1822009, -1.4669125, 4.0071440, -6.3098731, 7.6491127
1: -6.3575959, 9.4080048, -4.0064602, 6.2006745, -12.5582705, 13.4144650
2: -4.0071564, 8.5910988, -2.5055699, 5.5651913, -9.5723476, 11.0966682
3: -7.1218052, 10.4297237, -4.4741712, 6.7811985, -13.9030037, 14.9038944
4: -4.6281104, 10.4099588, -2.8958118, 6.7110949, -11.3392048, 13.3057709

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606282, upper bound: 17.9588331
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606502, upper bound: 17.9586389
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.2008598, 5.9014606, -1.6265631, 4.4408922, -6.6417522, 7.5280237
1: -6.0708818, 9.0127268, -4.4537382, 6.8536711, -12.9245529, 13.4664650
2: -3.8342478, 8.1894722, -2.7905805, 6.1733203, -10.0075684, 10.9800529
3: -6.8047171, 9.9628935, -4.9789329, 7.5049791, -14.3096962, 14.9418259
4: -4.4281912, 9.9270258, -3.2266197, 7.4491405, -11.8773317, 13.1536446

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601060, upper bound: 17.9619918
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9600474, upper bound: 17.9614913
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.3027291, 6.1822009, -1.6265631, 4.4408922, -6.7436213, 7.8087635
1: -6.3575959, 9.4080048, -4.4537382, 6.8536711, -13.2112675, 13.8617430
2: -4.0071564, 8.5910988, -2.7905805, 6.1733203, -10.1804771, 11.3816795
3: -7.1218052, 10.4297237, -4.9789329, 7.5049791, -14.6267843, 15.4086571
4: -4.6281104, 10.4099588, -3.2266197, 7.4491405, -12.0772495, 13.6365786

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607295, upper bound: 17.9619347
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -1.6045514, 4.3435383, -6.6283464, 7.7189813
1: -6.3056726, 9.3081608, -4.3935027, 6.6973906, -13.0030613, 13.7016640
2: -3.9664235, 8.4968262, -2.7491412, 6.0295582, -9.9959793, 11.2459679
3: -7.0619555, 10.3120298, -4.9114966, 7.3468657, -14.4088211, 15.2235241
4: -4.5765290, 10.2993307, -3.1771119, 7.2852683, -11.8617973, 13.4764423

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9569483
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9581323
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -1.7641354, 4.7743206, -7.0591297, 7.8785644
1: -6.3056726, 9.3081608, -4.8395271, 7.3438787, -13.6495514, 14.1476879
2: -3.9664235, 8.4968262, -3.0361805, 6.6364141, -10.6028376, 11.5330029
3: -7.0619555, 10.3120298, -5.4143553, 8.0728092, -15.1347637, 15.7263842
4: -4.5765290, 10.2993307, -3.5127008, 8.0279589, -12.6044874, 13.8120298

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9562877
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9619725
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -2.1817567, 5.8508301, -8.1356392, 8.2961864
1: -6.3056726, 9.3081608, -6.0132799, 8.9072704, -15.2129402, 15.3214407
2: -3.9664235, 8.4968262, -3.7834954, 8.1247444, -12.0911674, 12.2803211
3: -7.0619555, 10.3120298, -6.7310286, 9.8678417, -16.9297981, 17.0430565
4: -4.5765290, 10.2993307, -4.3695273, 9.8453188, -14.4218483, 14.6688576

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9565162
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9572709
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -2.3526335, 6.3142529, -8.5990620, 8.4670639
1: -6.3056726, 9.3081608, -6.4942160, 9.6076832, -15.9133539, 15.8023767
2: -3.9664235, 8.4968262, -4.0940924, 8.7726021, -12.7390251, 12.5909176
3: -7.0619555, 10.3120298, -7.2752800, 10.6551113, -17.7170677, 17.5873089
4: -4.5765290, 10.2993307, -4.7291746, 10.6296968, -15.2062263, 15.0285053

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9562568
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9612998
time: 0.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.34 seconds
NS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9560060
NS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9574641
NS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9560060
NS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605037, upper bound: 17.9574641
NS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609821
NS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609821
NS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609822
NS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9612348, upper bound: 17.9609822
NS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9570318, upper bound: 17.9557279
NS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9570318, upper bound: 17.9557279
NS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9570318, upper bound: 17.9559072
NS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9570319, upper bound: 17.9614494
NS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9610637, upper bound: 17.9584557
NS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9582616
NS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609762
NS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609762
NS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609761
NS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611064, upper bound: 17.9609761
NS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9559578, upper bound: 17.9552620
NS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9559578, upper bound: 17.9614540
NS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611313, upper bound: 17.9566023
NS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611313, upper bound: 17.9577862
NS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9600314, upper bound: 17.9555465
NS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9600314, upper bound: 17.9615401
NS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9556970
NS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9556970
NS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9556970
NS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9577265, upper bound: 17.9567111
NS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9448018
NS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9472009
NS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9613744, upper bound: 17.9510552
NS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9614167, upper bound: 17.9515266
NS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611245, upper bound: 17.9558464
NS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9611245, upper bound: 17.9567014
NS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9600393, upper bound: 17.9553769
NS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9600393, upper bound: 17.9605291
NS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9460534, upper bound: 17.9555116
NS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9460534, upper bound: 17.9612056
NS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9513839, upper bound: 17.9593479
NS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9513839, upper bound: 17.9610201
NS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9606282, upper bound: 17.9588331
NS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9606502, upper bound: 17.9586389
NS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9601060, upper bound: 17.9619918
NS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9600474, upper bound: 17.9614913
NS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9607295, upper bound: 17.9619347
NS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9569483
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9581323
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9562877
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9619725
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9565162
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9605859, upper bound: 17.9572709
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9562568
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.34
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9612998

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.5188491, 4.1307187, -1.5188491, 4.1307187, -5.6495676, 5.6495676
1: -4.1579542, 6.3804936, -4.1579542, 6.3804936, -10.5384464, 10.5384464
2: -2.5872672, 5.7286654, -2.5872672, 5.7286654, -8.3159323, 8.3159313
3: -4.6399579, 6.9638162, -4.6399579, 6.9638162, -11.6037741, 11.6037741
4: -2.9800746, 6.9294696, -2.9800746, 6.9294696, -9.9095421, 9.9095430

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9527782, upper bound: 17.9612364
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617392, upper bound: 17.9613562
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.5188491, 4.1307187, -1.5735397, 4.2779770, -5.7968264, 5.7042580
1: -4.1579542, 6.3804936, -4.3077784, 6.5949931, -10.7529459, 10.6882696
2: -2.5872672, 5.7286654, -2.6837308, 5.9328156, -8.5200825, 8.4123936
3: -4.6399579, 6.9638162, -4.8063774, 7.2089038, -11.8488607, 11.7701931
4: -2.9800746, 6.9294696, -3.0928497, 7.1728654, -10.1529369, 10.0223198

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9527782, upper bound: 17.9612364
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617392, upper bound: 17.9613562
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.5735397, 4.2779770, -1.5188491, 4.1307187, -5.7042580, 5.7968264
1: -4.3077784, 6.5949931, -4.1579542, 6.3804936, -10.6882696, 10.7529459
2: -2.6837308, 5.9328156, -2.5872672, 5.7286654, -8.4123955, 8.5200825
3: -4.8063774, 7.2089038, -4.6399579, 6.9638162, -11.7701931, 11.8488607
4: -3.0928497, 7.1728654, -2.9800746, 6.9294696, -10.0223179, 10.1529379

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9481286, upper bound: 17.9582288
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612286, upper bound: 17.9609802
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.5735397, 4.2779770, -1.5735397, 4.2779770, -5.8515167, 5.8515167
1: -4.3077784, 6.5949931, -4.3077784, 6.5949931, -10.9027691, 10.9027691
2: -2.6837308, 5.9328156, -2.6837308, 5.9328156, -8.6165466, 8.6165466
3: -4.8063774, 7.2089038, -4.8063774, 7.2089038, -12.0152817, 12.0152817
4: -3.0928497, 7.1728654, -3.0928497, 7.1728654, -10.2657127, 10.2657137

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9481286, upper bound: 17.9582288
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612286, upper bound: 17.9609802
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.5681590, 4.2715874, -1.7010814, 4.5886421, -6.1568003, 5.9726682
1: -4.2907004, 6.5990438, -4.6642113, 7.0726247, -11.3633251, 11.2632551
2: -2.6810930, 5.9358191, -2.9182312, 6.3778706, -9.0589638, 8.8540497
3: -4.7947497, 7.2176991, -5.2167773, 7.7601538, -12.5549030, 12.4344769
4: -3.0943124, 7.1687469, -3.3682301, 7.7145586, -10.8088713, 10.5369768

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604978, upper bound: 17.9557541
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9600477, upper bound: 17.9607195
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7190464, 4.6567822, -1.3811721, 3.7852652, -5.5043116, 6.0379539
1: -4.7160978, 7.1701775, -3.7716916, 5.8794641, -10.5955610, 10.9418688
2: -2.9575763, 6.4752569, -2.3550487, 5.2582474, -8.2158232, 8.8303051
3: -5.2760663, 7.8768725, -4.2116585, 6.4169946, -11.6930599, 12.0885296
4: -3.4220204, 7.8321195, -2.7208197, 6.3430138, -9.7650309, 10.5529366

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610637, upper bound: 17.9582615
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610637, upper bound: 17.9582616
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7190464, 4.6567822, -1.4413860, 3.9470685, -5.6661148, 6.0981679
1: -4.7160978, 7.1701775, -3.9359519, 6.1123199, -10.8284168, 11.1061296
2: -2.9575763, 6.4752569, -2.4611344, 5.4820280, -8.4396038, 8.9363909
3: -5.2760663, 7.8768725, -4.3944702, 6.6826639, -11.9587288, 12.2713413
4: -3.4220204, 7.8321195, -2.8455334, 6.6109185, -10.0329380, 10.6776524

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9582616
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9582616
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6740775, 4.5360212, -1.5351330, 4.2034473, -5.8775249, 6.0711541
1: -4.5909591, 6.9983420, -4.2014861, 6.5039902, -11.0949497, 11.1998281
2: -2.8773313, 6.3060794, -2.6298337, 5.8439541, -8.7212849, 8.9359131
3: -5.1356077, 7.6802788, -4.6960320, 7.1133051, -12.2489100, 12.3763103
4: -3.3281691, 7.6252813, -3.0404196, 7.0532660, -10.3814344, 10.6657009

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9525341, upper bound: 17.9606239
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614014, upper bound: 17.9609756
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6740775, 4.5360212, -1.6017711, 4.3825369, -6.0566144, 6.1377926
1: -4.5909591, 6.9983420, -4.3855538, 6.7691998, -11.3601589, 11.3838959
2: -2.8773313, 6.3060794, -2.7474403, 6.0930133, -8.9703426, 9.0535202
3: -5.1356077, 7.6802788, -4.9022079, 7.4093633, -12.5449705, 12.5824871
4: -3.3281691, 7.6252813, -3.1777031, 7.3523021, -10.6804695, 10.8029842

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9525341, upper bound: 17.9606239
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614014, upper bound: 17.9609756
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.7382356, 4.7122388, -1.5351330, 4.2034473, -5.9416828, 6.2473717
1: -4.7675924, 7.2522669, -4.2014861, 6.5039902, -11.2715826, 11.4537525
2: -2.9908900, 6.5507212, -2.6298337, 5.8439541, -8.8348446, 9.1805544
3: -5.3331223, 7.9705186, -4.6960320, 7.1133051, -12.4464264, 12.6665506
4: -3.4612842, 7.9243174, -3.0404196, 7.0532660, -10.5145502, 10.9647369

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607096, upper bound: 17.9604416
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607387, upper bound: 17.9606288
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7382356, 4.7122388, -1.6017711, 4.3825369, -6.1207724, 6.3140097
1: -4.7675924, 7.2522669, -4.3855538, 6.7691998, -11.5367928, 11.6378212
2: -2.9908900, 6.5507212, -2.7474403, 6.0930133, -9.0839024, 9.2981615
3: -5.3331223, 7.9705186, -4.9022079, 7.4093633, -12.7424850, 12.8727264
4: -3.4612842, 7.9243174, -3.1777031, 7.3523021, -10.8135843, 11.1020203

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607096, upper bound: 17.9604416
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607387, upper bound: 17.9606288
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.5651021, 4.2343812, -1.7010814, 4.5886421, -6.1537442, 5.9354620
1: -4.2856231, 6.5295439, -4.6642113, 7.0726247, -11.3582478, 11.1937532
2: -2.6765900, 5.8743706, -2.9182312, 6.3778706, -9.0544596, 8.7926006
3: -4.7896037, 7.1508865, -5.2167773, 7.7601538, -12.5497551, 12.3676624
4: -3.0873482, 7.0983520, -3.3682301, 7.7145586, -10.8019066, 10.4665823

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9525878, upper bound: 17.9550761
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9557825, upper bound: 17.9582710
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.4363624, 3.8960283, -5.5971098, 6.0250044
1: -4.6642113, 7.0726247, -3.9086945, 6.0137320, -10.6779404, 10.9813185
2: -2.9182312, 6.3778706, -2.4557543, 5.3967543, -8.3149853, 8.8336248
3: -5.2167773, 7.7601538, -4.3704510, 6.5981259, -11.8149023, 12.1306047
4: -3.3682301, 7.7145586, -2.8470242, 6.4974222, -9.8656521, 10.5615826

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598308, upper bound: 17.9560068
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9555038, upper bound: 17.9566022
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.5651021, 4.2343812, -5.9354620, 6.1537442
1: -4.6642113, 7.0726247, -4.2856231, 6.5295439, -11.1937542, 11.3582478
2: -2.9182312, 6.3778706, -2.6765900, 5.8743706, -8.7926006, 9.0544605
3: -5.2167773, 7.7601538, -4.7896037, 7.1508865, -12.3676624, 12.5497541
4: -3.3682301, 7.7145586, -3.0873482, 7.0983520, -10.4665823, 10.8019066

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598308, upper bound: 17.9571908
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611313, upper bound: 17.9577862
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.7010814, 4.5886421, -6.2897234, 6.2897234
1: -4.6642113, 7.0726247, -4.6642113, 7.0726247, -11.7368336, 11.7368345
2: -2.9182312, 6.3778706, -2.9182312, 6.3778706, -9.2961016, 9.2961016
3: -5.2167773, 7.7601538, -5.2167773, 7.7601538, -12.9769287, 12.9769297
4: -3.3682301, 7.7145586, -3.3682301, 7.7145586, -11.0827885, 11.0827885

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9591211, upper bound: 17.9552677
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9600314, upper bound: 17.9555465
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.6705869, 4.4756937, -6.1767750, 6.2592292
1: -4.6642113, 7.0726247, -4.5681796, 6.8725495, -11.5367594, 11.6408043
2: -2.9182312, 6.3778706, -2.8821464, 6.2137852, -9.1320162, 9.2600174
3: -5.2167773, 7.7601538, -5.1131873, 7.5820637, -12.7988405, 12.8733397
4: -3.3682301, 7.7145586, -3.3373265, 7.5138359, -10.8820658, 11.0518856

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9594873, upper bound: 17.9442064
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9448018
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.8139337, 4.8666945, -6.5677757, 6.4025760
1: -4.6642113, 7.0726247, -4.9869347, 7.4631629, -12.1273746, 12.0595579
2: -2.9182312, 6.3778706, -3.1260049, 6.7634478, -9.6816788, 9.5038757
3: -5.2167773, 7.7601538, -5.5779891, 8.2200403, -13.4368172, 13.3381405
4: -3.3682301, 7.7145586, -3.6069016, 8.1930723, -11.5613022, 11.3214607

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9594874, upper bound: 17.9466055
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9472009
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.9617697, 5.2819810, -6.9830623, 6.5504117
1: -4.6642113, 7.0726247, -5.4064646, 8.0788879, -12.7430992, 12.4790897
2: -2.9182312, 6.3778706, -3.3930345, 7.3429985, -10.2612295, 9.7709045
3: -5.2167773, 7.7601538, -6.0497794, 8.9081545, -14.1249304, 13.8099318
4: -3.3682301, 7.7145586, -3.9171157, 8.8984909, -12.2667208, 11.6316738

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609730, upper bound: 17.9508995
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609730, upper bound: 17.9510552
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -2.0242922, 5.4517026, -7.1527839, 6.6129332
1: -4.6642113, 7.0726247, -5.5753055, 8.3385353, -13.0027437, 12.6479292
2: -2.9182312, 6.3778706, -3.5078702, 7.5796757, -10.4979057, 9.8857393
3: -5.2167773, 7.7601538, -6.2422271, 9.2054749, -14.4222527, 14.0023794
4: -3.3682301, 7.7145586, -4.0535598, 9.1772156, -12.5454454, 11.7681179

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610153, upper bound: 17.9513709
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610153, upper bound: 17.9515266
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -1.9802357, 5.2870312, -6.9881120, 6.5688772
1: -4.6642113, 7.0726247, -5.4332342, 8.0638676, -12.7280769, 12.5058584
2: -2.9182312, 6.3778706, -3.4342861, 7.3374243, -10.2556553, 9.8121567
3: -5.2167773, 7.7601538, -6.0838666, 8.9362984, -14.1530743, 13.8440208
4: -3.3682301, 7.7145586, -3.9727602, 8.8857031, -12.2539330, 11.6873188

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601020, upper bound: 17.9555489
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611244, upper bound: 17.9558464
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7010814, 4.5886421, -2.1418462, 5.7338047, -7.4348860, 6.7304883
1: -4.6642113, 7.0726247, -5.9041319, 8.7336445, -13.3978548, 12.9767570
2: -2.9182312, 6.3778706, -3.7100067, 7.9628057, -10.8810358, 10.0878773
3: -5.2167773, 7.7601538, -6.6077490, 9.6664581, -14.8832340, 14.3679018
4: -3.3682301, 7.7145586, -4.2809086, 9.6543550, -13.0225849, 11.9954662

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601020, upper bound: 17.9563294
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611245, upper bound: 17.9567014
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.8139337, 4.8666945, -1.7010814, 4.5886421, -6.4025755, 6.5677757
1: -4.9869347, 7.4631629, -4.6642113, 7.0726247, -12.0595589, 12.1273746
2: -3.1260049, 6.7634478, -2.9182312, 6.3778706, -9.5038757, 9.6816788
3: -5.5779891, 8.2200403, -5.2167773, 7.7601538, -13.3381405, 13.4368172
4: -3.6069016, 8.1930723, -3.3682301, 7.7145586, -11.3214607, 11.5613022

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9455378, upper bound: 17.9585280
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9443073, upper bound: 17.9569666
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.0242922, 5.4517026, -1.7624259, 4.7696519, -6.7939439, 7.2141285
1: -5.5753055, 8.3385353, -4.8347473, 7.3370724, -12.9123783, 13.1732826
2: -3.5078702, 7.5796757, -3.0330470, 6.6300502, -10.1379204, 10.6127224
3: -6.2422271, 9.2054749, -5.4089136, 8.0649805, -14.3072052, 14.6143885
4: -4.0535598, 9.1772156, -3.5089464, 8.0202703, -12.0738297, 12.6861601

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.3027291, 6.1822009, -1.4413860, 3.9470685, -6.2497978, 7.6235862
1: -6.3575959, 9.4080048, -3.9359519, 6.1123199, -12.4699154, 13.3439569
2: -4.0071564, 8.5910988, -2.4611344, 5.4820280, -9.4891844, 11.0522327
3: -7.1218052, 10.4297237, -4.3944702, 6.6826639, -13.8044691, 14.8241930
4: -4.6281104, 10.4099588, -2.8455334, 6.6109185, -11.2390289, 13.2554922

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606502, upper bound: 17.9586023
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606502, upper bound: 17.9586389
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.2008598, 5.9014606, -1.5351330, 4.2034473, -6.4043069, 7.4365935
1: -6.0708818, 9.0127268, -4.2014861, 6.5039902, -12.5748720, 13.2142124
2: -3.8342478, 8.1894722, -2.6298337, 5.8439541, -9.6782017, 10.8193054
3: -6.8047171, 9.9628935, -4.6960320, 7.1133051, -13.9180222, 14.6589251
4: -4.4281912, 9.9270258, -3.0404196, 7.0532660, -11.4814568, 12.9674454

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9613906
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9614913
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.2008598, 5.9014606, -1.6017711, 4.3825369, -6.5833969, 7.5032320
1: -6.0708818, 9.0127268, -4.3855538, 6.7691998, -12.8400822, 13.3982811
2: -3.8342478, 8.1894722, -2.7474403, 6.0930133, -9.9272594, 10.9369125
3: -6.8047171, 9.9628935, -4.9022079, 7.4093633, -14.2140808, 14.8651009
4: -4.4281912, 9.9270258, -3.1777031, 7.3523021, -11.7804918, 13.1047287

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9613906
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9614913
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.3027291, 6.1822009, -1.5351330, 4.2034473, -6.5061765, 7.7173338
1: -6.3575959, 9.4080048, -4.2014861, 6.5039902, -12.8615856, 13.6094913
2: -4.0071564, 8.5910988, -2.6298337, 5.8439541, -9.8511105, 11.2209320
3: -7.1218052, 10.4297237, -4.6960320, 7.1133051, -14.2351103, 15.1257553
4: -4.6281104, 10.4099588, -3.0404196, 7.0532660, -11.6813755, 13.4503784

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.3027291, 6.1822009, -1.6017711, 4.3825369, -6.6852660, 7.7839718
1: -6.3575959, 9.4080048, -4.3855538, 6.7691998, -13.1267958, 13.7935581
2: -4.0071564, 8.5910988, -2.7474403, 6.0930133, -10.1001682, 11.3385391
3: -7.1218052, 10.4297237, -4.9022079, 7.4093633, -14.5311680, 15.3319321
4: -4.6281104, 10.4099588, -3.1777031, 7.3523021, -11.9804096, 13.5876617

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -1.7010814, 4.5886421, -6.8734503, 7.8155112
1: -6.3056726, 9.3081608, -4.6642113, 7.0726247, -13.3782959, 13.9723711
2: -3.9664235, 8.4968262, -2.9182312, 6.3778706, -10.3442936, 11.4150572
3: -7.0619555, 10.3120298, -5.2167773, 7.7601538, -14.8221083, 15.5288057
4: -4.5765290, 10.2993307, -3.3682301, 7.7145586, -12.2910881, 13.6675606

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9591228, upper bound: 17.9562877
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9618551
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -2.2848091, 6.1144300, -8.3992386, 8.3992386
1: -6.3056726, 9.3081608, -6.3056726, 9.3081608, -15.6138325, 15.6138325
2: -3.9664235, 8.4968262, -3.9664235, 8.4968262, -12.4632473, 12.4632473
3: -7.0619555, 10.3120298, -7.0619555, 10.3120298, -17.3739853, 17.3739853
4: -4.5765290, 10.2993307, -4.5765290, 10.2993307, -14.8758593, 14.8758593

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9592061, upper bound: 17.9562568
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9611715
time: 0.90 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.64 seconds
NS_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9527782, upper bound: 17.9612364
NS_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9617392, upper bound: 17.9613562
NS_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9527782, upper bound: 17.9612364
NS_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9617392, upper bound: 17.9613562
NS_A1_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9481286, upper bound: 17.9582288
NS_A1_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9612286, upper bound: 17.9609802
NS_A1_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9481286, upper bound: 17.9582288
NS_A1_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9612286, upper bound: 17.9609802
NS_A1_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9604978, upper bound: 17.9557541
NS_A1_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9600477, upper bound: 17.9607195
NS_A1_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9610637, upper bound: 17.9582615
NS_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9610637, upper bound: 17.9582616
NS_A1_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9582616
NS_A1_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9610857, upper bound: 17.9582616
NS_A1_B1_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9525341, upper bound: 17.9606239
NS_A1_B1_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9614014, upper bound: 17.9609756
NS_A1_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9525341, upper bound: 17.9606239
NS_A1_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9614014, upper bound: 17.9609756
NS_A1_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9607096, upper bound: 17.9604416
NS_A1_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9607387, upper bound: 17.9606288
NS_A1_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9607096, upper bound: 17.9604416
NS_A1_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9607387, upper bound: 17.9606288
NS_A1_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9525878, upper bound: 17.9550761
NS_A1_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9557825, upper bound: 17.9582710
NS_A1_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9598308, upper bound: 17.9560068
NS_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9555038, upper bound: 17.9566022
NS_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9598308, upper bound: 17.9571908
NS_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9611313, upper bound: 17.9577862
NS_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9591211, upper bound: 17.9552677
NS_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9600314, upper bound: 17.9555465
NS_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9594873, upper bound: 17.9442064
NS_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9448018
NS_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9594874, upper bound: 17.9466055
NS_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9608544, upper bound: 17.9472009
NS_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9609730, upper bound: 17.9508995
NS_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9609730, upper bound: 17.9510552
NS_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9610153, upper bound: 17.9513709
NS_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9610153, upper bound: 17.9515266
NS_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9601020, upper bound: 17.9555489
NS_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9611244, upper bound: 17.9558464
NS_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9601020, upper bound: 17.9563294
NS_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9611245, upper bound: 17.9567014
NS_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9455378, upper bound: 17.9585280
NS_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9443073, upper bound: 17.9569666
NS_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9606502, upper bound: 17.9586023
NS_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9606502, upper bound: 17.9586389
NS_A2_B1_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9613906
NS_A2_B1_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9614913
NS_A2_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9613906
NS_A2_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9554283, upper bound: 17.9614913
NS_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
NS_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
NS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
NS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9606709, upper bound: 17.9613191
NS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9591228, upper bound: 17.9562877
NS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9618551
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9592061, upper bound: 17.9562568
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 3, lower bound: -17.9598762, upper bound: 17.9611715

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.4735782, 4.0044985, -1.5188491, 4.1307187, -5.6042967, 5.5233469
1: -4.0310273, 6.1938324, -4.1579542, 6.3804936, -10.4115191, 10.3517866
2: -2.5059500, 5.5502505, -2.5872672, 5.7286654, -8.2346144, 8.1375179
3: -4.4968495, 6.7518468, -4.6399579, 6.9638162, -11.4606657, 11.3918047
4: -2.8844190, 6.7192111, -2.9800746, 6.9294696, -9.8138885, 9.6992855

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9527967, upper bound: 17.9527895
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9527967, upper bound: 17.9617260
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.5057840, 4.0987720, -1.5188491, 4.1307187, -5.6365027, 5.6176209
1: -4.1223822, 6.3348093, -4.1579542, 6.3804936, -10.5028725, 10.4927635
2: -2.5645714, 5.6856294, -2.5872672, 5.7286654, -8.2932367, 8.2728949
3: -4.6006532, 6.9117556, -4.6399579, 6.9638162, -11.5644693, 11.5517139
4: -2.9540472, 6.8770413, -2.9800746, 6.9294696, -9.8835163, 9.8571138

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617577, upper bound: 17.9529114
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617577, upper bound: 17.9618459
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.4735782, 4.0044985, -1.5735397, 4.2779770, -5.7515554, 5.5780373
1: -4.0310273, 6.1938324, -4.3077784, 6.5949931, -10.6260185, 10.5016108
2: -2.5059500, 5.5502505, -2.6837308, 5.9328156, -8.4387655, 8.2339802
3: -4.4968495, 6.7518468, -4.8063774, 7.2089038, -11.7057533, 11.5582237
4: -2.8844190, 6.7192111, -3.0928497, 7.1728654, -10.0572834, 9.8120613

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9524297, upper bound: 17.9484137
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9524297, upper bound: 17.9484137
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.5057840, 4.0987720, -1.5735397, 4.2779770, -5.7837610, 5.6723108
1: -4.1223822, 6.3348093, -4.3077784, 6.5949931, -10.7173719, 10.6425877
2: -2.5645714, 5.6856294, -2.6837308, 5.9328156, -8.4973869, 8.3693581
3: -4.6006532, 6.9117556, -4.8063774, 7.2089038, -11.8095570, 11.7181330
4: -2.9540472, 6.8770413, -3.0928497, 7.1728654, -10.1269112, 9.9698896

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613907, upper bound: 17.9485326
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613907, upper bound: 17.9613562
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.5600644, 4.2447262, -1.5188491, 4.1307187, -5.6907830, 5.7635751
1: -4.2710776, 6.5469871, -4.1579542, 6.3804936, -10.6515703, 10.7049408
2: -2.6602802, 5.8878298, -2.5872672, 5.7286654, -8.3889456, 8.4750967
3: -4.7657604, 7.1543503, -4.6399579, 6.9638162, -11.7295761, 11.7943058
4: -3.0659249, 7.1181812, -2.9800746, 6.9294696, -9.9953918, 10.0982552

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612471, upper bound: 17.9525369
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612471, upper bound: 17.9614709
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.5600644, 4.2447262, -1.5735397, 4.2779770, -5.8380413, 5.8182650
1: -4.2710776, 6.5469871, -4.3077784, 6.5949931, -10.8660698, 10.8547640
2: -2.6602802, 5.8878298, -2.6837308, 5.9328156, -8.5930958, 8.5715599
3: -4.7657604, 7.1543503, -4.8063774, 7.2089038, -11.9746647, 11.9607277
4: -3.0659249, 7.1181812, -3.0928497, 7.1728654, -10.2387877, 10.2110310

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608801, upper bound: 17.9483952
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608801, upper bound: 17.9609802
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.5438087, 4.2148871, -1.7010814, 4.5886421, -6.1324511, 5.9159679
1: -4.2236404, 6.5173674, -4.6642113, 7.0726247, -11.2962646, 11.1815748
2: -2.6387825, 5.8579068, -2.9182312, 6.3778706, -9.0166531, 8.7761374
3: -4.7192235, 7.1246295, -5.2167773, 7.7601538, -12.4793768, 12.3414059
4: -3.0464962, 7.0749168, -3.3682301, 7.7145586, -10.7610550, 10.4431467

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611143, upper bound: 17.9607195
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611143, upper bound: 17.9607195
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.6292620, 4.4194555, -1.3811721, 3.7852652, -5.4145269, 5.8006277
1: -4.4684100, 6.8257265, -3.7716916, 5.8794641, -10.3478718, 10.5974178
2: -2.7994349, 6.1463366, -2.3550487, 5.2582474, -8.0576811, 8.5013847
3: -4.9986148, 7.4858427, -4.2116585, 6.4169946, -11.4156094, 11.6974983
4: -3.2381997, 7.4308438, -2.7208197, 6.3430138, -9.5812130, 10.1516638

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9596578, upper bound: 17.9468833
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610484, upper bound: 17.9583822
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.6933944, 4.5952535, -1.3811721, 3.7852652, -5.4786596, 5.9764252
1: -4.6447144, 7.0794487, -3.7716916, 5.8794641, -10.5241776, 10.8511400
2: -2.9127038, 6.3903198, -2.3550487, 5.2582474, -8.1709499, 8.7453690
3: -5.1955366, 7.7756128, -4.2116585, 6.4169946, -11.6125298, 11.9872713
4: -3.3710847, 7.7293019, -2.7208197, 6.3430138, -9.7140980, 10.4501219

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9596578, upper bound: 17.9468833
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610484, upper bound: 17.9583822
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6292620, 4.4194555, -1.4413860, 3.9470685, -5.5763302, 5.8608418
1: -4.4684100, 6.8257265, -3.9359519, 6.1123199, -10.5807285, 10.7616777
2: -2.7994349, 6.1463366, -2.4611344, 5.4820280, -8.2814627, 8.6074705
3: -4.9986148, 7.4858427, -4.3944702, 6.6826639, -11.6812782, 11.8803091
4: -3.2381997, 7.4308438, -2.8455334, 6.6109185, -9.8491182, 10.2763767

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9596578, upper bound: 17.9533656
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610484, upper bound: 17.9582032
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6933944, 4.5952535, -1.4413860, 3.9470685, -5.6404629, 6.0366387
1: -4.6447144, 7.0794487, -3.9359519, 6.1123199, -10.7570333, 11.0154009
2: -2.9127038, 6.3903198, -2.4611344, 5.4820280, -8.3947306, 8.8514538
3: -5.1955366, 7.7756128, -4.3944702, 6.6826639, -11.8781986, 12.1700830
4: -3.3710847, 7.7293019, -2.8455334, 6.6109185, -9.9820032, 10.5748348

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9596578, upper bound: 17.9527985
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610484, upper bound: 17.9582032
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.6603758, 4.5025611, -1.5351330, 4.2034473, -5.8638229, 6.0376940
1: -4.5536509, 6.9505320, -4.2014861, 6.5039902, -11.0576410, 11.1520166
2: -2.8534312, 6.2608323, -2.6298337, 5.8439541, -8.6973858, 8.8906660
3: -5.0942330, 7.6256056, -4.6960320, 7.1133051, -12.2075367, 12.3216381
4: -3.3016722, 7.5704651, -3.0404196, 7.0532660, -10.3549385, 10.6108847

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613400, upper bound: 17.9523373
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613400, upper bound: 17.9616264
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.6603758, 4.5025611, -1.6017711, 4.3825369, -6.0429125, 6.1043320
1: -4.5536509, 6.9505320, -4.3855538, 6.7691998, -11.3228512, 11.3360844
2: -2.8534312, 6.2608323, -2.7474403, 6.0930133, -8.9464436, 9.0082722
3: -5.0942330, 7.6256056, -4.9022079, 7.4093633, -12.5035963, 12.5278130
4: -3.3016722, 7.5704651, -3.1777031, 7.3523021, -10.6539726, 10.7481680

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.6301234, 4.4781008, -1.5351330, 4.2034473, -5.8335705, 6.0132337
1: -4.4796209, 6.8908100, -4.2014861, 6.5039902, -10.9836111, 11.0922937
2: -2.8106396, 6.2244124, -2.6298337, 5.8439541, -8.6545935, 8.8542461
3: -5.0141931, 7.5631447, -4.6960320, 7.1133051, -12.1274948, 12.2591763
4: -3.2593732, 7.5374107, -3.0404196, 7.0532660, -10.3126392, 10.5778294

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9606416, upper bound: 17.9519410
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607603, upper bound: 17.9610531
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7119666, 4.6495304, -1.5351330, 4.2034473, -5.9154134, 6.1846633
1: -4.6966786, 7.1604486, -4.2014861, 6.5039902, -11.2006683, 11.3619347
2: -2.9457979, 6.4641514, -2.6298337, 5.8439541, -8.7897520, 9.0939827
3: -5.2540169, 7.8665400, -4.6960320, 7.1133051, -12.3673220, 12.5625725
4: -3.4099405, 7.8204226, -3.0404196, 7.0532660, -10.4632063, 10.8608408

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606700, upper bound: 17.9522706
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607887, upper bound: 17.9613505
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6301234, 4.4781008, -1.6017711, 4.3825369, -6.0126600, 6.0798721
1: -4.4796209, 6.8908100, -4.3855538, 6.7691998, -11.2488203, 11.2763624
2: -2.8106396, 6.2244124, -2.7474403, 6.0930133, -8.9036512, 8.9718523
3: -5.0141931, 7.5631447, -4.9022079, 7.4093633, -12.4235544, 12.4653530
4: -3.2593732, 7.5374107, -3.1777031, 7.3523021, -10.6116753, 10.7151136

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602289, upper bound: 17.9604234
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602289, upper bound: 17.9604416
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7119666, 4.6495304, -1.6017711, 4.3825369, -6.0945034, 6.2513018
1: -4.6966786, 7.1604486, -4.3855538, 6.7691998, -11.4658785, 11.5460024
2: -2.9457979, 6.4641514, -2.7474403, 6.0930133, -9.0388088, 9.2115889
3: -5.2540169, 7.8665400, -4.9022079, 7.4093633, -12.6633797, 12.7687473
4: -3.4099405, 7.8204226, -3.1777031, 7.3523021, -10.7622423, 10.9981251

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602482, upper bound: 17.9605737
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9602482, upper bound: 17.9606288
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6551759, 4.4693856, -1.5651021, 4.2343812, -5.8895569, 6.0344877
1: -4.5387487, 6.8951001, -4.2856231, 6.5295439, -11.0682926, 11.1807232
2: -2.8383393, 6.2138143, -2.6765900, 5.8743706, -8.7127094, 8.8904037
3: -5.0763092, 7.5605025, -4.7896037, 7.1508865, -12.2271957, 12.3501043
4: -3.2757318, 7.5150108, -3.0873482, 7.0983520, -10.3740835, 10.6023579

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608965, upper bound: 17.9568558
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609371, upper bound: 17.9575116
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.6551759, 4.4693856, -1.6705869, 4.4756937, -6.1308694, 6.1399727
1: -4.5387487, 6.8951001, -4.5681796, 6.8725495, -11.4112988, 11.4632797
2: -2.8383393, 6.2138143, -2.8821464, 6.2137852, -9.0521240, 9.0959606
3: -5.0763092, 7.5605025, -5.1131873, 7.5820637, -12.6583719, 12.6736889
4: -3.2757318, 7.5150108, -3.3373265, 7.5138359, -10.7895679, 10.8523359

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9600984, upper bound: 17.9418735
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603495, upper bound: 17.9441779
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6551759, 4.4693856, -1.8139337, 4.8666945, -6.5218701, 6.2833195
1: -4.5387487, 6.8951001, -4.9869347, 7.4631629, -12.0019112, 11.8820343
2: -2.8383393, 6.2138143, -3.1260049, 6.7634478, -9.6017876, 9.3398190
3: -5.0763092, 7.5605025, -5.5779891, 8.2200403, -13.2963495, 13.1384888
4: -3.2757318, 7.5150108, -3.6069016, 8.1930723, -11.4688044, 11.1219110

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607202, upper bound: 17.9466043
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605533, upper bound: 17.9453739
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.6464120, 4.4400134, -1.9617697, 5.2819810, -6.9283929, 6.4017825
1: -4.5170956, 6.8415451, -5.4064646, 8.0788879, -12.5959835, 12.2480087
2: -2.8197491, 6.1677041, -3.3930345, 7.3429985, -10.1627474, 9.5607386
3: -5.0482073, 7.4958010, -6.0497794, 8.9081545, -13.9563608, 13.5455780
4: -3.2475328, 7.4630890, -3.9171157, 8.8984909, -12.1460238, 11.3802052

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.6987832, 4.5823746, -1.9617697, 5.2819810, -6.9807644, 6.5441442
1: -4.6578503, 7.0633750, -5.4064646, 8.0788879, -12.7367382, 12.4698391
2: -2.9140451, 6.3691845, -3.3930345, 7.3429985, -10.2570438, 9.7622185
3: -5.2095246, 7.7494984, -6.0497794, 8.9081545, -14.1176796, 13.7992764
4: -3.3631778, 7.7040763, -3.9171157, 8.8984909, -12.2616692, 11.6211901

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6464120, 4.4400134, -2.0242922, 5.4517026, -7.0981145, 6.4643040
1: -4.5170956, 6.8415451, -5.5753055, 8.3385353, -12.8556309, 12.4168482
2: -2.8197491, 6.1677041, -3.5078702, 7.5796757, -10.3994246, 9.6755733
3: -5.0482073, 7.4958010, -6.2422271, 9.2054749, -14.2536821, 13.7380257
4: -3.2475328, 7.4630890, -4.0535598, 9.1772156, -12.4247475, 11.5166492

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6987832, 4.5823746, -2.0242922, 5.4517026, -7.1504860, 6.6066656
1: -4.6578503, 7.0633750, -5.5753055, 8.3385353, -12.9963856, 12.6386795
2: -2.9140451, 6.3691845, -3.5078702, 7.5796757, -10.4937181, 9.8770542
3: -5.2095246, 7.7494984, -6.2422271, 9.2054749, -14.4149990, 13.9917240
4: -3.3631778, 7.7040763, -4.0535598, 9.1772156, -12.5403938, 11.7576342

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.6551759, 4.4693856, -1.9802357, 5.2870312, -6.9422064, 6.4496212
1: -4.5387487, 6.8951001, -5.4332342, 8.0638676, -12.6026154, 12.3283348
2: -2.8383393, 6.2138143, -3.4342861, 7.3374243, -10.1757631, 9.6480999
3: -5.0763092, 7.5605025, -6.0838666, 8.9362984, -14.0126066, 13.6443691
4: -3.2757318, 7.5150108, -3.9727602, 8.8857031, -12.1614351, 11.4877710

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601599, upper bound: 17.9544109
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601599, upper bound: 17.9558464
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6551759, 4.4693856, -2.1418462, 5.7338047, -7.3889809, 6.6112318
1: -4.5387487, 6.8951001, -5.9041319, 8.7336445, -13.2723932, 12.7992315
2: -2.8383393, 6.2138143, -3.7100067, 7.9628057, -10.8011446, 9.9238205
3: -5.0763092, 7.5605025, -6.6077490, 9.6664581, -14.7427654, 14.1682501
4: -3.2757318, 7.5150108, -4.2809086, 9.6543550, -12.9300871, 11.7959166

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603172, upper bound: 17.9547317
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603173, upper bound: 17.9567014
time: 0.58 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.83 + 418.29 = 421.12 seconds
