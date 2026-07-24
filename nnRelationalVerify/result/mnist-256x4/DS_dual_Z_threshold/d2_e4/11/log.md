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
execution time: IAR + RelationalAnalysis = 2.10 + 10.47 = 12.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732361
time: 5.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732361
time: 6.28 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.16 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.16
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732361
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.16
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732361

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3732353, upper bound: 157.3732361
time: 6.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732353
time: 6.76 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3732353, upper bound: 157.3732361
time: 5.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732353
time: 6.36 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.28 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.28
Output dim: 9, lower bound: -157.3732353, upper bound: 157.3732361
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.28
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732353
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.28
Output dim: 9, lower bound: -157.3732353, upper bound: 157.3732361
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.28
Output dim: 9, lower bound: -157.3732361, upper bound: 157.3732353

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
time: 5.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
time: 5.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786
time: 5.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
time: 5.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
time: 5.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786
time: 5.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786
time: 5.49 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.11 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502786, upper bound: 157.3502789
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.11
Output dim: 9, lower bound: -157.3502789, upper bound: 157.3502786

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 5.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 4.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 4.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 5.03 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 16.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 5.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 5.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 5.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 6.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 5.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 6.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 5.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 6.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 6.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 5.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
time: 4.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 5.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 5.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 5.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 5.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 5.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 5.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
time: 6.43 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.51
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 6.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 4.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 6.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 5.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 6.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 4.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
time: 5.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
time: 4.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
time: 5.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
time: 6.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
time: 5.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.9962311, 69.7704010, -86.9962311, 69.7704010, -156.7666168, 156.7666168
1: -73.2806931, 62.2428398, -73.2806931, 62.2428398, -135.5235138, 135.5235138
2: -96.2169037, 63.0441971, -96.2169037, 63.0441971, -159.2611084, 159.2611084
3: -102.1828690, 54.5574303, -102.1828690, 54.5574303, -156.7402802, 156.7402802
4: -93.7343674, 72.0458450, -93.7343674, 72.0458450, -165.7802124, 165.7802124
5: -84.0396576, 66.0810089, -84.0396576, 66.0810089, -150.1206665, 150.1206665
6: -80.4295883, 77.6040802, -80.4295883, 77.6040802, -158.0336456, 158.0336456
7: -87.6262131, 73.8585892, -87.6262131, 73.8585892, -161.4848022, 161.4848022
8: -105.5955505, 72.3908997, -105.5955505, 72.3908997, -177.9864349, 177.9864349
9: -79.7351074, 78.8443680, -79.7351074, 78.8443680, -158.5794678, 158.5794678

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Candidate
type: DSZ, layer: 1, pos: 85

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
time: 6.02 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 13.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202294, upper bound: 157.3202292
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202293
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202293, upper bound: 157.3202292
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 9, lower bound: -157.3202292, upper bound: 157.3202294
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279224, upper bound: 157.3279252
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279231, upper bound: 157.3279252
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279231
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 9, lower bound: -157.3279252, upper bound: 157.3279224

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 12.58 + 592.59 = 605.17 seconds
