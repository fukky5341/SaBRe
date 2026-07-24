## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 157.221417204


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168)
1: (-73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138)
2: (-96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084)
3: (-102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802)
4: (-93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124)
5: (-84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665)
6: (-80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456)
7: (-87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022)
8: (-105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349)
9: (-79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.03 + 10.31 = 12.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3737930, upper bound: 157.3734072
time: 10.07 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960
time: 8.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 18.46 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 18.46
Output dim: 9, lower bound: -157.3737930, upper bound: 157.3734072
NS_A2, status: Status.UNKNOWN, split count: 1, time: 18.46
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -91.0355301, 72.9609985, -86.5079727, 69.3835220, -160.4190521, 159.4689636
1: -76.7187500, 65.1230011, -72.8738403, 61.9002914, -138.6190033, 137.9968262
2: -100.7244720, 65.9272614, -95.6827545, 62.6966705, -163.4211426, 161.6100006
3: -106.9438553, 57.0885468, -101.6076508, 54.2569466, -161.2008057, 158.6961975
4: -98.0582886, 75.3670883, -93.2079391, 71.6474380, -169.7057037, 168.5750275
5: -87.9225082, 69.0830460, -83.5727615, 65.7176590, -153.6401520, 152.6557922
6: -84.1449890, 81.2097855, -79.9794388, 77.1732025, -161.3181763, 161.1891937
7: -91.6963806, 77.2472000, -87.1392441, 73.4500580, -165.1464233, 164.3864441
8: -110.4620514, 75.6930847, -105.0050964, 71.9860687, -182.4480896, 180.6981812
9: -83.3841782, 82.4561234, -79.2906952, 78.4063644, -161.7905426, 161.7468109

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3543192, upper bound: 157.3546307
time: 8.05 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3468404, upper bound: 157.3471472
time: 7.35 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3538055, upper bound: 157.3537977
time: 8.44 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3723603, upper bound: 157.3721446
time: 7.41 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3722741, upper bound: 157.3720539
time: 7.55 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -86.8164749, 69.6279068, -86.9962311, 69.7704010, -156.5868683, 156.6240997
1: -73.1305084, 62.1165543, -73.2806931, 62.2428398, -135.3733215, 135.3972473
2: -96.0200806, 62.9162636, -96.2169037, 63.0441971, -159.0642700, 159.1331482
3: -101.9706116, 54.4467583, -102.1828690, 54.5574303, -156.5279999, 156.6296082
4: -93.5400772, 71.8985519, -93.7343674, 72.0458450, -165.5859222, 165.6329193
5: -83.8675537, 65.9471207, -84.0396576, 66.0810089, -149.9485626, 149.9867859
6: -80.2637329, 77.4450836, -80.4295883, 77.6040802, -157.8677826, 157.8746490
7: -87.4466782, 73.7080307, -87.6262131, 73.8585892, -161.3052673, 161.3342438
8: -105.3775177, 72.2411880, -105.5955505, 72.3908997, -177.7684174, 177.8367157
9: -79.5709076, 78.6828537, -79.7351074, 78.8443680, -158.4152832, 158.4179535

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3662091, upper bound: 157.3651466
time: 7.46 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3627165, upper bound: 157.3627165
time: 6.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.78 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 15.78
Output dim: 9, lower bound: -157.3723603, upper bound: 157.3721446
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 15.78
Output dim: 9, lower bound: -157.3722741, upper bound: 157.3720539
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.78
Output dim: 9, lower bound: -157.3662091, upper bound: 157.3651466
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.78
Output dim: 9, lower bound: -157.3627165, upper bound: 157.3627165

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -86.0162506, 68.9763412, -84.7408829, 67.9803238, -153.9965820, 153.7172089
1: -72.4296570, 61.5167389, -71.3636932, 60.6306267, -133.0602722, 132.8804321
2: -95.1175079, 62.3266258, -93.7089462, 61.4292374, -156.5467529, 156.0355682
3: -100.9823227, 53.9297829, -99.5085678, 53.1455383, -154.1278687, 153.4383545
4: -92.6440811, 71.2143173, -91.3022385, 70.1852951, -162.8293762, 162.5165558
5: -83.0917664, 65.3229065, -81.8722534, 64.3932114, -147.4849854, 147.1951447
6: -79.4767532, 76.7188187, -78.3360748, 75.5922470, -155.0689850, 155.0549011
7: -86.5908127, 73.0005646, -85.3418732, 71.9552307, -158.5460510, 158.3424072
8: -104.3630066, 71.5802002, -102.8581772, 70.5375595, -174.9005737, 174.4383698
9: -78.8109512, 77.9262619, -77.6811676, 76.8110580, -155.6220093, 155.6074219

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3596796, upper bound: 157.3586454
time: 7.10 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 7.32 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -87.2658539, 69.9541931, -83.9080048, 67.3158188, -154.5816650, 153.8621674
1: -73.4523544, 62.3742332, -70.6566772, 60.0352249, -133.4875793, 133.0309143
2: -96.4679947, 63.2056122, -92.7819977, 60.8345909, -157.3025818, 155.9875336
3: -102.4236984, 54.6726341, -98.5193710, 52.6229591, -155.0466461, 153.1919708
4: -93.9906845, 72.2240219, -90.4023972, 69.4988098, -163.4894714, 162.6264038
5: -84.2771759, 66.2553558, -81.0673218, 63.7713852, -148.0485535, 147.3226624
6: -80.6192856, 77.8100815, -77.5656738, 74.8476715, -155.4669495, 155.3757324
7: -87.8356705, 74.0533218, -84.5006409, 71.2557678, -159.0914307, 158.5539551
8: -105.8412476, 72.5897064, -101.8453903, 69.8500977, -175.6913452, 174.4350891
9: -79.9593048, 79.0307388, -76.9264832, 76.0614090, -156.0206757, 155.9572144

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3615538, upper bound: 157.3615693
time: 7.41 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3578959, upper bound: 157.3571613
time: 7.47 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -86.8164749, 69.6279068, -86.5251999, 69.3973770, -156.2138519, 156.1530914
1: -73.1305084, 62.1165543, -72.8867645, 61.9130058, -135.0434875, 135.0033264
2: -96.0200806, 62.9162636, -95.7007980, 62.7084885, -158.7285614, 158.6170654
3: -101.9706116, 54.4467583, -101.6292267, 54.2671432, -156.2377167, 156.0759735
4: -93.5400772, 71.8985519, -93.2279892, 71.6605682, -165.2006531, 165.1265411
5: -83.8675537, 65.9471207, -83.5874023, 65.7291489, -149.5966797, 149.5345154
6: -80.2637329, 77.4450836, -79.9972229, 77.1888199, -157.4525299, 157.4423065
7: -87.4466782, 73.7080307, -87.1546478, 73.4621201, -160.9087982, 160.8626709
8: -105.3775177, 72.2411880, -105.0305939, 72.0056229, -177.3831482, 177.2717743
9: -79.5709076, 78.6828537, -79.3041229, 78.4218521, -157.9927673, 157.9869690

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3627165, upper bound: 157.3627165
time: 7.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3627165, upper bound: 157.3627165
time: 6.58 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -86.2266541, 69.1608429, -101.1407089, 80.9892273, -167.2158813, 170.3015442
1: -72.6369095, 61.7032585, -85.1796799, 72.2621765, -144.8990784, 146.8829193
2: -95.3729553, 62.4954987, -111.7848053, 73.0577774, -168.4307251, 174.2803040
3: -101.2753525, 54.0829887, -118.8056335, 63.3240051, -164.5993500, 172.8886261
4: -92.9053650, 71.4151535, -108.8764725, 83.5516891, -176.4570465, 180.2915955
5: -83.3013382, 65.5065155, -97.5829239, 76.5055084, -159.8068237, 163.0893860
6: -79.7213593, 76.9249802, -93.4788361, 90.1295929, -169.8509369, 170.4038086
7: -86.8556519, 73.2110443, -101.7534332, 85.6234360, -172.4790955, 174.9644775
8: -104.6693115, 71.7584915, -122.6836777, 84.0042343, -188.6735535, 194.4421692
9: -79.0311661, 78.1534271, -92.4529114, 91.4933777, -170.5245361, 170.6063385

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509155, upper bound: 157.3502253
time: 7.15 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3487430, upper bound: 157.3487430
time: 6.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.77 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3596796, upper bound: 157.3586454
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3615538, upper bound: 157.3615693
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3578959, upper bound: 157.3571613
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3627165, upper bound: 157.3627165
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3627165, upper bound: 157.3627165
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3509155, upper bound: 157.3502253
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 15.77
Output dim: 9, lower bound: -157.3487430, upper bound: 157.3487430

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -86.0047226, 68.9672089, -84.0968094, 67.4697952, -153.4745178, 153.0640106
1: -72.4200516, 61.5086670, -70.8262253, 60.1792908, -132.5993347, 132.3348541
2: -95.1048355, 62.3184433, -93.0010452, 60.9721832, -156.0770264, 155.3194580
3: -100.9688721, 53.9227066, -98.7568359, 52.7497215, -153.7185974, 152.6795349
4: -92.6316528, 71.2048798, -90.6080780, 69.6577988, -162.2894592, 161.8129272
5: -83.0807190, 65.3142624, -81.2559128, 63.9113235, -146.9920349, 146.5701599
6: -79.4661789, 76.7086487, -77.7451401, 75.0232391, -154.4893646, 154.4537964
7: -86.5792542, 72.9908676, -84.6963272, 71.4131241, -157.9923706, 157.6871948
8: -104.3491211, 71.5707550, -102.0826263, 70.0098572, -174.3589783, 173.6533813
9: -78.8003540, 77.9159241, -77.0894775, 76.2328262, -155.0331726, 155.0054016

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 7.89 seconds

## Relational analysis of NS_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 7.24 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -85.3121719, 68.4189072, -91.2112732, 73.0956421, -158.4078064, 159.6301575
1: -71.8423767, 61.0237350, -76.8092499, 65.1899185, -137.0322876, 137.8329468
2: -94.3437347, 61.8282051, -100.8509521, 66.0167847, -160.3605194, 162.6791534
3: -100.1625061, 53.4980736, -107.1298294, 57.1252937, -157.2877960, 160.6278839
4: -91.8863907, 70.6384888, -98.2239609, 75.4880371, -167.3744202, 168.8624573
5: -82.4182434, 64.7967606, -88.1173019, 69.2116013, -151.6298370, 152.9140625
6: -78.8319778, 76.0966415, -84.2296524, 81.3202133, -160.1521759, 160.3262482
7: -85.8862915, 72.4091415, -91.8297577, 77.3193588, -163.2056427, 164.2388916
8: -103.5167999, 71.0047760, -110.6890106, 75.9318314, -179.4486084, 181.6937866
9: -78.1655273, 77.2955399, -83.4941788, 82.4970093, -160.6625214, 160.7897186

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 7.29 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 6.72 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: -86.6202164, 69.4428787, -83.8965302, 67.3067093, -153.9269257, 153.3394012
1: -72.9143066, 61.9220505, -70.6470947, 60.0271683, -132.9414673, 132.5691528
2: -95.7588272, 62.7480850, -92.7693634, 60.8264503, -156.5852814, 155.5174103
3: -101.6712723, 54.2767105, -98.5059586, 52.6159058, -154.2871704, 152.7826538
4: -93.2960052, 71.6957474, -90.3900375, 69.4893951, -162.7854004, 162.0857391
5: -83.6598511, 65.7722321, -81.0563049, 63.7627945, -147.4226227, 146.8285370
6: -80.0279922, 77.2397766, -77.5551300, 74.8375168, -154.8655090, 154.7948761
7: -87.1895676, 73.5105286, -84.4891281, 71.2461014, -158.4356689, 157.9996643
8: -105.0650558, 72.0611877, -101.8315582, 69.8406982, -174.9057617, 173.8927307
9: -79.3665771, 78.4518661, -76.9159393, 76.0510864, -155.4176636, 155.3677979

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_A1_A2_A1_B1

### Relational analysis result of NS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3578959, upper bound: 157.3571613
time: 6.68 seconds

## Relational analysis of NS_A1_A2_A1_B2

### Relational analysis result of NS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3578959, upper bound: 157.3571613
time: 6.38 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -93.8032608, 75.1224594, -83.1999512, 66.7545547, -160.5578156, 158.3224030
1: -78.9564438, 66.9818420, -70.0652771, 59.5393753, -138.4958038, 137.0471039
2: -103.6819077, 67.8403473, -92.0033798, 60.3329430, -164.0148468, 159.8437042
3: -110.1267242, 58.6969032, -97.6938171, 52.1881485, -162.3148651, 156.3907166
4: -100.9866486, 77.5789719, -89.6390686, 68.9192200, -169.9058685, 167.2180023
5: -90.5883408, 71.1197968, -80.3892746, 63.2423630, -153.8307037, 151.5090637
6: -86.5747299, 83.5969086, -76.9161606, 74.2219849, -160.7966766, 160.5130615
7: -94.3908081, 79.4724426, -83.7910995, 70.6602402, -165.0510559, 163.2635498
8: -113.7526016, 78.0384445, -100.9930496, 69.2708206, -183.0234070, 179.0314789
9: -85.8300552, 84.7763901, -76.2767563, 75.4260941, -161.2561493, 161.0531311

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_A2_A2_A1

### Relational analysis result of NS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3491238, upper bound: 157.3489956
time: 7.22 seconds

## Relational analysis of NS_A1_A2_A2_A2

### Relational analysis result of NS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3401682, upper bound: 157.3396829
time: 6.09 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -86.3453979, 69.2548218, -86.5251999, 69.3973770, -155.7427673, 155.7800293
1: -72.7365265, 61.7867012, -72.8867645, 61.9130058, -134.6495361, 134.6734619
2: -95.5039062, 62.5805168, -95.7007980, 62.7084885, -158.2124023, 158.2813110
3: -101.4168777, 54.1564560, -101.6292267, 54.2671432, -155.6839905, 155.7856750
4: -93.0336838, 71.5132217, -93.2279892, 71.6605682, -164.6942291, 164.7412109
5: -83.4152222, 65.5952072, -83.5874023, 65.7291489, -149.1443787, 149.1826019
6: -79.8313370, 77.0298004, -79.9972229, 77.1888199, -157.0201569, 157.0270233
7: -86.9750443, 73.3115234, -87.1546478, 73.4621201, -160.4371643, 160.4661713
8: -104.8125305, 71.8558960, -105.0305939, 72.0056229, -176.8181458, 176.8864899
9: -79.1399078, 78.2603226, -79.3041229, 78.4218521, -157.5617676, 157.5644379

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3420095, upper bound: 157.3420160
time: 8.97 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3362138, upper bound: 157.3353738
time: 7.78 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -100.9673386, 80.8517532, -86.5251999, 69.3973770, -170.3647003, 167.3769379
1: -85.0349503, 72.1402740, -72.8867645, 61.9130058, -146.9479370, 145.0270386
2: -111.5949936, 72.9342880, -95.7007980, 62.7084885, -174.3034821, 168.6350861
3: -118.6010818, 63.2173042, -101.6292267, 54.2671432, -172.8682251, 164.8465118
4: -108.6894302, 83.4098053, -93.2279892, 71.6605682, -180.3499908, 176.6377869
5: -97.4170151, 76.3762283, -83.5874023, 65.7291489, -163.1461487, 159.9636230
6: -93.3190155, 89.9761810, -79.9972229, 77.1888199, -170.5078430, 169.9734039
7: -101.5805130, 85.4783478, -87.1546478, 73.4621201, -175.0426331, 172.6329956
8: -122.4736557, 83.8597870, -105.0305939, 72.0056229, -194.4792786, 188.8903809
9: -92.2947617, 91.3375854, -79.3041229, 78.4218521, -170.7166138, 170.6417084

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3542699, upper bound: 157.3539209
time: 8.01 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3522406, upper bound: 157.3512706
time: 7.20 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -86.2266541, 69.1608429, -97.8427811, 78.3466873, -164.5733337, 167.0036316
1: -72.6369095, 61.7032585, -82.3712082, 69.8901367, -142.5270386, 144.0744629
2: -95.3729553, 62.4954987, -108.0902328, 70.6751709, -166.0481262, 170.5857239
3: -101.2753525, 54.0829887, -114.8928452, 61.2587814, -162.5341034, 168.9758301
4: -92.9053650, 71.4151535, -105.3182755, 80.8087006, -173.7140350, 176.7333832
5: -83.3013382, 65.5065155, -94.3826370, 74.0140686, -157.3153992, 159.8891144
6: -79.7213593, 76.9249802, -90.4165344, 87.1660843, -166.8874512, 167.3414917
7: -86.8556519, 73.2110443, -98.4167480, 82.8251495, -169.6808014, 171.6277924
8: -104.6693115, 71.7584915, -118.6738586, 81.2508850, -185.9201965, 190.4323425
9: -79.0311661, 78.1534271, -89.4416733, 88.5010681, -167.5322266, 167.5950928

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3453604, upper bound: 157.3446491
time: 7.24 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3452731, upper bound: 157.3445601
time: 6.56 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -82.4261246, 66.1273193, -94.4282074, 75.5996246, -158.0257568, 160.5554962
1: -69.4006271, 58.9786377, -79.4031906, 67.3857422, -136.7863770, 138.3818054
2: -91.1255188, 59.7576675, -104.1861572, 68.1645737, -159.2901001, 163.9438171
3: -96.7698593, 51.7096367, -110.7717438, 59.1023369, -155.8721924, 162.4813385
4: -88.8099136, 68.2627029, -101.6269226, 77.9210052, -166.7308960, 169.8896179
5: -79.6233521, 62.6508751, -91.0352325, 71.4459534, -151.0692902, 153.6860657
6: -76.1988754, 73.5203857, -87.2211685, 84.0691681, -160.2680359, 160.7415466
7: -83.0157089, 69.9966583, -94.9290390, 79.9139862, -162.9296875, 164.9256744
8: -100.0606689, 68.5984879, -114.5017853, 78.3682251, -178.4288940, 183.1002808
9: -75.5681229, 74.7206268, -86.3593063, 85.4107132, -160.9788361, 161.0799255

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3434051, upper bound: 157.3432207
time: 6.81 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3427843, upper bound: 157.3427843
time: 4.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 13.19 seconds
NS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3578959, upper bound: 157.3571613
NS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3578959, upper bound: 157.3571613
NS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3491238, upper bound: 157.3489956
NS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3401682, upper bound: 157.3396829
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3420095, upper bound: 157.3420160
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3362138, upper bound: 157.3353738
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3542699, upper bound: 157.3539209
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3522406, upper bound: 157.3512706
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3453604, upper bound: 157.3446491
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3452731, upper bound: 157.3445601
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3434051, upper bound: 157.3432207
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 9, lower bound: -157.3427843, upper bound: 157.3427843

## BFS NS instance: NS_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -85.3661270, 68.4615936, -84.0968094, 67.4697952, -152.8359070, 152.5583954
1: -71.8877640, 61.0613518, -70.8262253, 60.1792908, -132.0670471, 131.8875580
2: -94.4033737, 61.8658028, -93.0010452, 60.9721832, -155.3755493, 154.8668518
3: -100.2244644, 53.5309181, -98.7568359, 52.7497215, -152.9741821, 152.2877502
4: -91.9443741, 70.6822739, -90.6080780, 69.6577988, -161.6021729, 161.2903290
5: -82.4701385, 64.8365707, -81.2559128, 63.9113235, -146.3814697, 146.0924835
6: -78.8811951, 76.1445389, -77.7451401, 75.0232391, -153.9044037, 153.8896790
7: -85.9400101, 72.4540329, -84.6963272, 71.4131241, -157.3531342, 157.1503601
8: -103.5811996, 71.0482025, -102.0826263, 70.0098572, -173.5910645, 173.1308289
9: -78.2143326, 77.3435059, -77.0894775, 76.2328262, -154.4471588, 154.4329834

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: B, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3596796, upper bound: 157.3586454
time: 5.89 seconds

## Relational analysis of NS_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3596796, upper bound: 157.3586454
time: 6.98 seconds

## BFS NS instance: NS_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -92.5177917, 74.1139908, -84.0968094, 67.4697952, -159.9875641, 158.2107849
1: -77.8998032, 66.0989456, -70.8262253, 60.1792908, -138.0790863, 136.9251251
2: -102.2914581, 66.9347534, -93.0010452, 60.9721832, -163.2636414, 159.9357910
3: -108.6457367, 57.9324493, -98.7568359, 52.7497215, -161.3954315, 156.6892853
4: -99.6019592, 76.5404358, -90.6080780, 69.6577988, -169.2597656, 167.1485138
5: -89.3653107, 70.1602707, -81.2559128, 63.9113235, -153.2766418, 151.4161835
6: -85.3986206, 82.4737549, -77.7451401, 75.0232391, -160.4218292, 160.2188873
7: -93.1107712, 78.3898468, -84.6963272, 71.4131241, -164.5238953, 163.0861816
8: -112.2300491, 76.9957581, -102.0826263, 70.0098572, -182.2398987, 179.0783844
9: -84.6524200, 83.6396332, -77.0894775, 76.2328262, -160.8852539, 160.7290955

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of NS_A1_A1_B1_A2_A1

### Relational analysis result of NS_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3519716, upper bound: 157.3517119
time: 7.71 seconds

## Relational analysis of NS_A1_A1_B1_A2_A2

### Relational analysis result of NS_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3422540, upper bound: 157.3414666
time: 7.82 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -85.3121719, 68.4189072, -87.7891006, 70.3773727, -155.6895142, 156.2079773
1: -71.8423767, 61.0237350, -73.8838959, 62.7313080, -134.5736847, 134.9076080
2: -94.3437347, 61.8282051, -97.0271606, 63.5618019, -157.9055328, 158.8553619
3: -100.1625061, 53.4980736, -103.0665207, 54.9722557, -155.1347656, 156.5645905
4: -91.8863907, 70.6384888, -94.5330582, 72.6571045, -164.5434570, 165.1715393
5: -82.4182434, 64.7967606, -84.8230057, 66.6473846, -149.0656281, 149.6197662
6: -78.8319778, 76.0966415, -81.0476990, 78.2570953, -157.0890808, 157.1443329
7: -85.8862915, 72.4091415, -88.3496017, 74.4251633, -160.3114319, 160.7587433
8: -103.5167999, 71.0047760, -106.5309219, 73.1264648, -176.6432648, 177.5357056
9: -78.1655273, 77.2955399, -80.3780975, 79.4079590, -157.5734711, 157.6736450

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 7.26 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 7.62 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -85.3121719, 68.4189072, -89.0381317, 71.3539505, -156.6661224, 157.4570312
1: -71.8423767, 61.0237350, -74.9063263, 63.5883560, -135.4307251, 135.9300537
2: -94.3437347, 61.8282051, -98.3770218, 64.4408417, -158.7845459, 160.2052307
3: -100.1625061, 53.4980736, -104.5116653, 55.7144966, -155.8769989, 158.0097351
4: -91.8863907, 70.6384888, -95.8792725, 73.6649475, -165.5513306, 166.5177460
5: -82.4182434, 64.7967606, -86.0086517, 67.5773010, -149.9955139, 150.8054199
6: -78.8319778, 76.0966415, -82.1894073, 79.3456650, -158.1776428, 158.2860260
7: -85.8862915, 72.4091415, -89.5939026, 75.4726639, -161.3589478, 162.0030060
8: -103.5167999, 71.0047760, -108.0098419, 74.1405411, -177.6573334, 179.0146179
9: -78.1655273, 77.2955399, -81.5220795, 80.5066605, -158.6721802, 158.8176270

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 7.31 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
time: 6.66 seconds

## BFS NS instance: NS_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -86.6202164, 69.4428787, -83.2636566, 66.8050156, -153.4252319, 152.7065277
1: -72.9143066, 61.9220505, -70.1189346, 59.5836830, -132.4979858, 132.0409851
2: -95.7588272, 62.7480850, -92.0737076, 60.3773193, -156.1361389, 154.8217621
3: -101.6712723, 54.2767105, -97.7672806, 52.2269363, -153.8982086, 152.0439911
4: -93.2960052, 71.6957474, -89.7078705, 68.9710312, -162.2670135, 161.4036255
5: -83.6598511, 65.7722321, -80.4506531, 63.2892494, -146.9490662, 146.2228851
6: -80.0279922, 77.2397766, -76.9744263, 74.2783737, -154.3063660, 154.2142029
7: -87.1895676, 73.5105286, -83.8547745, 70.7134018, -157.9029541, 157.3652954
8: -105.0650558, 72.0611877, -101.0694427, 69.3220978, -174.3871460, 173.1306152
9: -79.3665771, 78.4518661, -76.3345184, 75.4828644, -154.8494415, 154.7863770

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: B, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_A2_A1_B1_A1

### Relational analysis result of NS_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3562810, upper bound: 157.3566715
time: 8.03 seconds

## Relational analysis of NS_A1_A2_A1_B1_A2

### Relational analysis result of NS_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3615538, upper bound: 157.3615693
time: 7.16 seconds

## BFS NS instance: NS_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -86.6202164, 69.4428787, -90.4127045, 72.4592896, -159.0794983, 159.8555908
1: -72.9143066, 61.9220505, -76.1305923, 64.6187439, -137.5330505, 138.0526428
2: -95.7588272, 62.7480850, -99.9623413, 65.4460068, -161.2048340, 162.7104034
3: -101.6712723, 54.2767105, -106.1827087, 56.6224899, -158.2937469, 160.4593964
4: -93.2960052, 71.6957474, -97.3623505, 74.8290405, -168.1250153, 169.0581055
5: -83.6598511, 65.7722321, -87.3464355, 68.6145172, -152.2743683, 153.1186676
6: -80.0279922, 77.2397766, -83.4912796, 80.6063995, -160.6343994, 160.7310486
7: -87.1895676, 73.5105286, -91.0235596, 76.6478653, -163.8374329, 164.5340881
8: -105.0650558, 72.0611877, -109.7179565, 75.2733231, -180.3383789, 181.7791443
9: -79.3665771, 78.4518661, -82.7695389, 81.7785873, -161.1451569, 161.2214050

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_A2_A1_B2_B1

### Relational analysis result of NS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3615539, upper bound: 157.3615693
time: 6.98 seconds

## Relational analysis of NS_A1_A2_A1_B2_B2

### Relational analysis result of NS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3615538, upper bound: 157.3615693
time: 6.08 seconds

## BFS NS instance: NS_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -91.3245773, 73.1497269, -82.7275848, 66.3786545, -157.7032166, 155.8773193
1: -76.8315506, 65.1943817, -69.6604843, 59.1992378, -136.0307617, 134.8548584
2: -100.8979874, 66.0405502, -91.4728470, 59.9902725, -160.8882599, 157.5133667
3: -107.1760788, 57.1309967, -97.1314011, 51.8901482, -159.0661926, 154.2623901
4: -98.3159103, 75.5038147, -89.1295853, 68.5240784, -166.8399963, 164.6333923
5: -88.1765366, 69.2389374, -79.9297638, 62.8843765, -151.0609131, 149.1686859
6: -84.2605057, 81.3755569, -76.4752808, 73.7988434, -158.0593567, 157.8508301
7: -91.8664398, 77.3576660, -83.3097229, 70.2575378, -162.1239624, 160.6673737
8: -110.7308807, 75.9740753, -100.4169846, 68.8777313, -179.6086121, 176.3910522
9: -83.5530319, 82.5373764, -75.8427811, 74.9996185, -158.5525970, 158.3801422

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_A2_A2_A1_A1

### Relational analysis result of NS_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3476596, upper bound: 157.3475456
time: 7.18 seconds

## Relational analysis of NS_A1_A2_A2_A1_A2

### Relational analysis result of NS_A1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3485645, upper bound: 157.3484368
time: 7.47 seconds

## BFS NS instance: NS_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -92.2039566, 73.8343658, -81.3321228, 65.2663040, -157.4702606, 155.1664886
1: -77.5207596, 65.7705688, -68.4687424, 58.1990089, -135.7197571, 134.2392883
2: -101.8133698, 66.6162033, -89.9111633, 58.9797401, -160.7931061, 156.5273285
3: -108.1612167, 57.6391144, -95.4766693, 51.0162849, -159.1774750, 153.1157532
4: -99.2477798, 76.1586990, -87.6253433, 67.3570328, -166.6048126, 163.7840424
5: -88.9851456, 69.8639069, -78.5718765, 61.8274422, -150.8125916, 148.4357605
6: -85.0247040, 82.1302567, -75.1737671, 72.5524979, -157.5771942, 157.3040161
7: -92.6996689, 78.0704117, -81.8924866, 69.0704269, -161.7700806, 159.9628906
8: -111.7478180, 76.6495667, -98.7182846, 67.7127838, -179.4606018, 175.3678589
9: -84.3438416, 83.3157883, -74.5630188, 73.7417221, -158.0855713, 157.8787994

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_A1_A2_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3401682, upper bound: 157.3396829
time: 6.03 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3401682, upper bound: 157.3396829
time: 7.02 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -85.1672668, 68.3107986, -86.5251999, 69.3973770, -154.5646362, 154.8359680
1: -71.7396774, 60.9430237, -72.8867645, 61.9130058, -133.6526794, 133.8297729
2: -94.1945114, 61.7395973, -95.7007980, 62.7084885, -156.9029999, 157.4403992
3: -100.0238342, 53.4194794, -101.6292267, 54.2671432, -154.2909851, 155.0487061
4: -91.7656631, 70.5409164, -93.2279892, 71.6605682, -163.4262390, 163.7689056
5: -82.2738800, 64.7093124, -83.5874023, 65.7291489, -148.0030212, 148.2967224
6: -78.7418900, 75.9730148, -79.9972229, 77.1888199, -155.9307098, 155.9702454
7: -85.7894135, 72.3177109, -87.1546478, 73.4621201, -159.2515259, 159.4723511
8: -103.3810959, 70.8730850, -105.0305939, 72.0056229, -175.3867035, 175.9036865
9: -78.0686493, 77.1969604, -79.3041229, 78.4218521, -156.4905090, 156.5010681

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3484318, upper bound: 157.3487582
time: 7.69 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3450403, upper bound: 157.3458057
time: 6.53 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -78.1261749, 62.6333427, -83.8793030, 67.2771683, -145.4033203, 146.5126190
1: -65.7240982, 55.8546219, -70.6473999, 60.0218582, -125.7459564, 126.5020065
2: -86.3263626, 56.6997871, -92.7627029, 60.8184853, -147.1448517, 149.4624786
3: -91.6861877, 49.0103378, -98.5062408, 52.6199646, -144.3061523, 147.5165710
4: -84.1845322, 64.7217407, -90.3781662, 69.4775085, -153.6620483, 155.0999146
5: -75.4082260, 59.4025536, -81.0210419, 63.7408562, -139.1490784, 140.4235992
6: -72.2106094, 69.6135559, -77.5514450, 74.8193893, -147.0299988, 147.1650085
7: -78.6937180, 66.3780823, -84.4927597, 71.2312393, -149.9249420, 150.8708038
8: -94.8068619, 64.9630508, -101.8191605, 69.7976456, -164.6045074, 166.7822113
9: -71.7058945, 70.8310852, -76.9012833, 76.0386581, -147.7445526, 147.7323608

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3435784, upper bound: 157.3432064
time: 8.07 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3417613, upper bound: 157.3417612
time: 5.34 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -97.6694412, 78.2092133, -86.5251999, 69.3973770, -167.0668182, 164.7344055
1: -82.2265320, 69.7682571, -72.8867645, 61.9130058, -144.1395416, 142.6550293
2: -107.9004593, 70.5517349, -95.7007980, 62.7084885, -170.6089478, 166.2525330
3: -114.6883011, 61.1520882, -101.6292267, 54.2671432, -168.9554443, 162.7812958
4: -105.1312866, 80.6668243, -93.2279892, 71.6605682, -176.7918549, 173.8948059
5: -94.2167740, 73.8848419, -83.5874023, 65.7291489, -159.9459229, 157.4722443
6: -90.2567215, 87.0127258, -79.9972229, 77.1888199, -167.4455414, 167.0099487
7: -98.2438812, 82.6801071, -87.1546478, 73.4621201, -171.7059937, 169.8347473
8: -118.4638748, 81.1064606, -105.0305939, 72.0056229, -190.4694824, 186.1370544
9: -89.2835617, 88.3453140, -79.3041229, 78.4218521, -167.7054138, 167.6494141

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3493274, upper bound: 157.3489309
time: 8.74 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3492552, upper bound: 157.3488212
time: 7.10 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -94.2552490, 75.4624329, -82.7201462, 66.3602905, -160.6155090, 158.1825409
1: -79.2588577, 67.2642365, -69.6466675, 59.1852188, -138.4440765, 136.9109039
2: -103.9967728, 68.0415039, -91.4484940, 59.9674187, -163.9641571, 159.4899902
3: -110.5676346, 58.9959106, -97.1184921, 51.8910522, -162.4586792, 156.1144104
4: -101.4403458, 77.7794876, -89.1276703, 68.5044861, -169.9448090, 166.9071503
5: -90.8697510, 71.3170242, -79.9050369, 62.8701439, -153.7398987, 151.2220612
6: -87.0617371, 83.9160919, -76.4706421, 73.7802811, -160.8420105, 160.3867340
7: -94.7565536, 79.7692871, -83.3101883, 70.2439194, -165.0004578, 163.0794220
8: -114.2922440, 78.2241058, -100.4165421, 68.8419724, -183.1342163, 178.6406555
9: -86.2015305, 85.2551651, -75.8369751, 74.9851303, -161.1866608, 161.0921326

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3473334, upper bound: 157.3464373
time: 8.11 seconds

## Relational analysis of NS_A2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3465205, upper bound: 157.3454650
time: 8.26 seconds

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -84.4613495, 67.7590714, -92.9226379, 74.4389648, -158.9003143, 160.6816864
1: -71.1282196, 60.4348183, -78.1670456, 66.3560410, -137.4842529, 138.6018677
2: -93.4011307, 61.2293358, -102.5902328, 67.1431046, -160.5442352, 163.8195648
3: -99.1783066, 52.9725800, -109.0572510, 58.1683960, -157.3466949, 162.0298309
4: -91.0015793, 69.9543762, -100.0103760, 76.7372208, -167.7388000, 169.9647522
5: -81.6025543, 64.1833191, -89.6464920, 70.3260880, -151.9286346, 153.8298035
6: -78.0795898, 75.3456650, -85.8427048, 82.7629623, -160.8425293, 161.1883392
7: -85.0600433, 71.7177277, -93.4121857, 78.6619110, -163.7219543, 165.1299133
8: -102.5246811, 70.3113937, -112.6952667, 77.2166290, -179.7412415, 183.0066528
9: -77.4231873, 76.5597534, -84.9597549, 84.0605774, -161.4837646, 161.5195007

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B2_B1_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3385853, upper bound: 157.3378360
time: 6.87 seconds

## Relational analysis of NS_A2_B2_B1_B1_B2

### Relational analysis result of NS_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3376803, upper bound: 157.3371087
time: 6.61 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -83.6276093, 67.0939102, -94.1018066, 75.3628006, -158.9904175, 161.1957092
1: -70.4205704, 59.8389359, -79.1260223, 67.1603622, -137.5809326, 138.9649506
2: -92.4732971, 60.6341667, -103.8589096, 67.9730759, -160.4463806, 164.4930573
3: -98.1881332, 52.4496307, -110.4153595, 58.8662148, -157.0543213, 162.8649902
4: -90.1008377, 69.2672882, -101.2808609, 77.6894226, -167.7902527, 170.5481415
5: -80.7968140, 63.5609703, -90.7632828, 71.2075500, -152.0043488, 154.3242493
6: -77.3084564, 74.6003494, -86.9181061, 83.7885284, -161.0969849, 161.5184479
7: -84.2180710, 71.0176620, -94.5814514, 79.6532745, -163.8713379, 165.5990906
8: -101.5109024, 69.6233902, -114.0933151, 78.1777496, -179.6886597, 183.7167053
9: -76.6678314, 75.8093796, -86.0444946, 85.0999756, -161.7678070, 161.8538818

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B2_B1_B2_B1

### Relational analysis result of NS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3384969, upper bound: 157.3377306
time: 6.94 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2

### Relational analysis result of NS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3376717, upper bound: 157.3370795
time: 6.60 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -80.6767883, 64.7385025, -89.6696854, 71.8201370, -152.4969177, 154.4081726
1: -67.9052887, 57.7216034, -75.3372955, 63.9681168, -131.8734131, 133.0588837
2: -89.1711884, 58.5027657, -98.8645782, 64.7464523, -153.9176331, 157.3673248
3: -94.6918640, 50.6089363, -105.1265106, 56.1128654, -150.8047028, 155.7354126
4: -86.9228668, 66.8154221, -96.4935532, 73.9821777, -160.9050446, 163.3089447
5: -77.9397812, 61.3405571, -86.4532318, 67.8785629, -145.8183289, 147.7937775
6: -74.5716324, 71.9554214, -82.7950439, 79.8111496, -154.3827820, 154.7504578
7: -81.2360916, 68.5167084, -90.0885391, 75.8841858, -157.1202698, 158.6052551
8: -97.9355774, 67.1652145, -108.7194748, 74.4657974, -172.4013672, 175.8846893
9: -73.9746475, 73.1417007, -82.0219498, 81.1162872, -155.0909424, 155.1636505

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3302343, upper bound: 157.3295534
time: 6.63 seconds

## Relational analysis of NS_A2_B2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3283644, upper bound: 157.3281447
time: 6.77 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -79.8150330, 64.0508041, -90.3955688, 72.3834686, -152.1985016, 154.4463654
1: -67.1735458, 57.1057701, -75.9119720, 64.4469604, -131.6205139, 133.0177002
2: -88.2130356, 57.8873444, -99.6324234, 65.2499619, -153.4629669, 157.5197449
3: -93.6694565, 50.0694847, -105.9507141, 56.5282440, -150.1976929, 156.0201569
4: -85.9925842, 66.1054764, -97.2739182, 74.5598297, -160.5524139, 163.3793945
5: -77.1077805, 60.6965485, -87.1362534, 68.4195175, -145.5272522, 147.8327637
6: -73.7758102, 71.1849136, -83.4514389, 80.4310760, -154.2068787, 154.6363525
7: -80.3661270, 67.7933426, -90.7990799, 76.4933624, -156.8594971, 158.5924225
8: -96.8873215, 66.4538193, -109.5666504, 75.0506439, -171.9379578, 176.0204773
9: -73.1935425, 72.3666382, -82.6924973, 81.7453995, -154.9389191, 155.0591278

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of NS_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3361817, upper bound: 157.3359201
time: 5.70 seconds

## Relational analysis of NS_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3350599, upper bound: 157.3350599
time: 5.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 16.62 seconds
NS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3596796, upper bound: 157.3586454
NS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3596796, upper bound: 157.3586454
NS_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3519716, upper bound: 157.3517119
NS_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3422540, upper bound: 157.3414666
NS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3582530, upper bound: 157.3575712
NS_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3562810, upper bound: 157.3566715
NS_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3615538, upper bound: 157.3615693
NS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3615539, upper bound: 157.3615693
NS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3615538, upper bound: 157.3615693
NS_A1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3476596, upper bound: 157.3475456
NS_A1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3485645, upper bound: 157.3484368
NS_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3401682, upper bound: 157.3396829
NS_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3401682, upper bound: 157.3396829
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3484318, upper bound: 157.3487582
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3450403, upper bound: 157.3458057
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3435784, upper bound: 157.3432064
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3417613, upper bound: 157.3417612
NS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3493274, upper bound: 157.3489309
NS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3492552, upper bound: 157.3488212
NS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3473334, upper bound: 157.3464373
NS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3465205, upper bound: 157.3454650
NS_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3385853, upper bound: 157.3378360
NS_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3376803, upper bound: 157.3371087
NS_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3384969, upper bound: 157.3377306
NS_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3376717, upper bound: 157.3370795
NS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3302343, upper bound: 157.3295534
NS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3283644, upper bound: 157.3281447
NS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3361817, upper bound: 157.3359201
NS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 16.62
Output dim: 9, lower bound: -157.3350599, upper bound: 157.3350599

## BFS NS instance: NS_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -85.3661270, 68.4615936, -80.7174072, 64.7866287, -150.1527557, 149.1790009
1: -71.8877640, 61.0613518, -67.9379349, 57.7520943, -129.6398621, 128.9992828
2: -94.4033737, 61.8658028, -89.2268143, 58.5489616, -152.9523163, 151.0926208
3: -100.2244644, 53.5309181, -94.7451477, 50.6247177, -150.8491669, 148.2760620
4: -91.9443741, 70.6822739, -86.9630508, 66.8636856, -158.8080597, 157.6453094
5: -82.4701385, 64.8365707, -78.0033417, 61.3800316, -143.8501740, 142.8399048
6: -78.8811951, 76.1445389, -74.6037979, 72.0001526, -150.8813324, 150.7483215
7: -85.9400101, 72.4540329, -81.2603149, 68.5552597, -154.4952698, 153.7143402
8: -103.5811996, 71.0482025, -97.9772110, 67.2410049, -170.8222046, 169.0254211
9: -78.2143326, 77.3435059, -74.0115509, 73.1835785, -151.3978882, 151.3550568

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3716456, upper bound: 157.3714525
time: 7.19 seconds

## Relational analysis of NS_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3716456, upper bound: 157.3716696
time: 7.27 seconds

## BFS NS instance: NS_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -85.3661270, 68.4615936, -81.9995422, 65.7900314, -151.1561584, 150.4611359
1: -71.8877640, 61.0613518, -68.9888458, 58.6307144, -130.5184784, 130.0502014
2: -94.4033737, 61.8658028, -90.6108170, 59.4509087, -153.8542786, 152.4766235
3: -100.2244644, 53.5309181, -96.2263260, 51.3891068, -151.6135712, 149.7572479
4: -91.9443741, 70.6822739, -88.3412323, 67.8990479, -159.8433990, 159.0234985
5: -82.4701385, 64.8365707, -79.2188873, 62.3365936, -144.8067322, 144.0554504
6: -78.8811951, 76.1445389, -75.7733688, 73.1184616, -151.9996490, 151.9178772
7: -85.9400101, 72.4540329, -82.5346451, 69.6336441, -155.5736542, 154.9886780
8: -103.5811996, 71.0482025, -99.4953690, 68.2799377, -171.8611450, 170.5435638
9: -78.2143326, 77.3435059, -75.1884766, 74.3146591, -152.5289764, 152.5319824

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 69

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 176

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3715316, upper bound: 157.3713021
time: 8.78 seconds

## Relational analysis of NS_A1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3723603, upper bound: 157.3721446
time: 7.76 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 12.34 + 603.75 = 616.09 seconds
