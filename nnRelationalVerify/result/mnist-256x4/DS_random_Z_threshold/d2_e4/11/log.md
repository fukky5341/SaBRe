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
execution time: IAR + RelationalAnalysis = 0.88 + 10.35 = 11.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -157.3787960, upper bound: 157.3787960

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179
time: 7.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179
time: 7.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.40 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.40
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.40
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3778995, upper bound: 157.3779031
time: 7.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3779030, upper bound: 157.3778997
time: 7.27 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179
time: 7.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179
time: 7.39 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.06 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.06
Output dim: 9, lower bound: -157.3778995, upper bound: 157.3779031
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.06
Output dim: 9, lower bound: -157.3779030, upper bound: 157.3778997
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.06
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.06
Output dim: 9, lower bound: -157.3783179, upper bound: 157.3783179

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3600093, upper bound: 157.3600088
time: 5.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3600093, upper bound: 157.3600088
time: 5.83 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3721886, upper bound: 157.3721897
time: 7.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3721886, upper bound: 157.3721897
time: 7.37 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3736243, upper bound: 157.3736236
time: 7.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3736243, upper bound: 157.3736236
time: 7.85 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3756468, upper bound: 157.3756499
time: 7.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3756468, upper bound: 157.3756499
time: 7.17 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 15.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3600093, upper bound: 157.3600088
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3600093, upper bound: 157.3600088
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3721886, upper bound: 157.3721897
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3721886, upper bound: 157.3721897
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3736243, upper bound: 157.3736236
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3736243, upper bound: 157.3736236
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3756468, upper bound: 157.3756499
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.22
Output dim: 9, lower bound: -157.3756468, upper bound: 157.3756499

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3576542, upper bound: 157.3576409
time: 6.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3576370, upper bound: 157.3576586
time: 6.91 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487251
time: 5.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487251
time: 5.84 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
time: 4.75 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
time: 4.75 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3690492, upper bound: 157.3690481
time: 6.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3690492, upper bound: 157.3690481
time: 6.80 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662039
time: 6.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662039
time: 6.26 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3697696, upper bound: 157.3697698
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3697696, upper bound: 157.3697698
time: 8.32 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3756436, upper bound: 157.3756499
time: 7.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3756468, upper bound: 157.3756441
time: 6.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 15.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3576542, upper bound: 157.3576409
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3576370, upper bound: 157.3576586
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487251
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487251
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509139
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3690492, upper bound: 157.3690481
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3690492, upper bound: 157.3690481
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662039
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662039
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3697696, upper bound: 157.3697698
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3697696, upper bound: 157.3697698
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3756436, upper bound: 157.3756499
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.15
Output dim: 9, lower bound: -157.3756468, upper bound: 157.3756441

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3408710, upper bound: 157.3408697
time: 6.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3408710, upper bound: 157.3408697
time: 5.67 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3576370, upper bound: 157.3576583
time: 6.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3576369, upper bound: 157.3576586
time: 7.49 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487242
time: 6.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487251
time: 5.56 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3485983, upper bound: 157.3485984
time: 6.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3485983, upper bound: 157.3485984
time: 6.40 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3403257, upper bound: 157.3403296
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3403257, upper bound: 157.3403296
time: 5.06 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509106, upper bound: 157.3509136
time: 6.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509134
time: 6.13 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509100, upper bound: 157.3509139
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509138
time: 5.69 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3310495, upper bound: 157.3310504
time: 5.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3310495, upper bound: 157.3310504
time: 5.33 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3566661, upper bound: 157.3566655
time: 7.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3566661, upper bound: 157.3566655
time: 7.83 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3521230, upper bound: 157.3521187
time: 5.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3521230, upper bound: 157.3521187
time: 5.61 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3662079, upper bound: 157.3662039
time: 7.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662039
time: 6.75 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3662058, upper bound: 157.3662039
time: 7.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662021
time: 6.17 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3685452, upper bound: 157.3685517
time: 7.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3685452, upper bound: 157.3685517
time: 7.02 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3686452, upper bound: 157.3686517
time: 6.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3686488, upper bound: 157.3686483
time: 8.40 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3755641, upper bound: 157.3755762
time: 7.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3755641, upper bound: 157.3755762
time: 7.67 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3647087, upper bound: 157.3647108
time: 6.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3647087, upper bound: 157.3647108
time: 6.55 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 13.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3408710, upper bound: 157.3408697
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3408710, upper bound: 157.3408697
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3576370, upper bound: 157.3576583
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3576369, upper bound: 157.3576586
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487242
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487251
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3485983, upper bound: 157.3485984
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3485983, upper bound: 157.3485984
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3403257, upper bound: 157.3403296
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3403257, upper bound: 157.3403296
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3509106, upper bound: 157.3509136
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509134
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3509100, upper bound: 157.3509139
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509138
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3310495, upper bound: 157.3310504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3310495, upper bound: 157.3310504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3566661, upper bound: 157.3566655
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3566661, upper bound: 157.3566655
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3521230, upper bound: 157.3521187
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3521230, upper bound: 157.3521187
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3662079, upper bound: 157.3662039
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662039
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3662058, upper bound: 157.3662039
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662021
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3685452, upper bound: 157.3685517
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3685452, upper bound: 157.3685517
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3686452, upper bound: 157.3686517
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3686488, upper bound: 157.3686483
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3755641, upper bound: 157.3755762
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3755641, upper bound: 157.3755762
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3647087, upper bound: 157.3647108
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 9, lower bound: -157.3647087, upper bound: 157.3647108

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3355097, upper bound: 157.3355107
time: 5.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3355097, upper bound: 157.3355107
time: 5.45 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3298945, upper bound: 157.3298879
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3298945, upper bound: 157.3298879
time: 5.06 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3453487, upper bound: 157.3453531
time: 5.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3453487, upper bound: 157.3453531
time: 5.94 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3439753, upper bound: 157.3439848
time: 7.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3439753, upper bound: 157.3439848
time: 7.20 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3359054, upper bound: 157.3359119
time: 5.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3359054, upper bound: 157.3359119
time: 5.85 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487248
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3487224, upper bound: 157.3487251
time: 6.37 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3468411, upper bound: 157.3468322
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3468286, upper bound: 157.3468458
time: 5.37 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3485983, upper bound: 157.3485983
time: 5.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3485978, upper bound: 157.3485984
time: 5.99 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3321861, upper bound: 157.3321860
time: 5.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3321861, upper bound: 157.3321860
time: 6.18 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3296836, upper bound: 157.3296845
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3296836, upper bound: 157.3296845
time: 5.22 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3479578, upper bound: 157.3479563
time: 7.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3479518, upper bound: 157.3479625
time: 5.60 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3485328, upper bound: 157.3485326
time: 7.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -157.3485328, upper bound: 157.3485326
time: 6.10 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 16.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3355097, upper bound: 157.3355107
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3355097, upper bound: 157.3355107
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3298945, upper bound: 157.3298879
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3298945, upper bound: 157.3298879
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3453487, upper bound: 157.3453531
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3453487, upper bound: 157.3453531
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3439753, upper bound: 157.3439848
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3439753, upper bound: 157.3439848
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3359054, upper bound: 157.3359119
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3359054, upper bound: 157.3359119
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3487237, upper bound: 157.3487248
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3487224, upper bound: 157.3487251
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3468411, upper bound: 157.3468322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3468286, upper bound: 157.3468458
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3485983, upper bound: 157.3485983
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3485978, upper bound: 157.3485984
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3321861, upper bound: 157.3321860
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3321861, upper bound: 157.3321860
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3296836, upper bound: 157.3296845
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3296836, upper bound: 157.3296845
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3479578, upper bound: 157.3479563
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3479518, upper bound: 157.3479625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3485328, upper bound: 157.3485326
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.32
Output dim: 9, lower bound: -157.3485328, upper bound: 157.3485326
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3509100, upper bound: 157.3509139
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3509107, upper bound: 157.3509138
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3310495, upper bound: 157.3310504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3310495, upper bound: 157.3310504
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3566661, upper bound: 157.3566655
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3566661, upper bound: 157.3566655
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3521230, upper bound: 157.3521187
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3521230, upper bound: 157.3521187
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3662079, upper bound: 157.3662039
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662039
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3662058, upper bound: 157.3662039
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3662082, upper bound: 157.3662021
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3685452, upper bound: 157.3685517
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3685452, upper bound: 157.3685517
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3686452, upper bound: 157.3686517
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3686488, upper bound: 157.3686483
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3755641, upper bound: 157.3755762
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3755641, upper bound: 157.3755762
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3647087, upper bound: 157.3647108
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.32
Output dim: 9, lower bound: -157.3647087, upper bound: 157.3647108

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.24 + 598.40 = 609.64 seconds
