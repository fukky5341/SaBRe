## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 385.180259218
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591)
1: (-210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100)
2: (-276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429)
3: (-294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862)
4: (-269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690)
5: (-241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797)
6: (-230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052)
7: (-251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947)
8: (-303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061)
9: (-228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316)

## BASE Result
execution time: IAR + LP analysis = 1.10 + 10.69 = 11.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -385.1965521, upper bound: 385.1965521


# Binary Search by BASE starts (time budget: 2688.22 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=387.1527099609375
rel_dist={1: [-385.1965121761082, 385.19651217610806]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=387.1527099609375
rel_dist={1: [-385.1964844738563, 385.19648447385634]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=387.1527099609375
rel_dist={1: [-385.1964274816625, 385.19642747694013]}

## Binary Search Result
Binary search time: 43.17 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2645.05 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910525, upper bound: 385.1910525
time: 9.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910525, upper bound: 385.1910525
time: 8.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.25
Output dim: 1, lower bound: -385.1910525, upper bound: 385.1910525
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.25
Output dim: 1, lower bound: -385.1910525, upper bound: 385.1910525

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910520, upper bound: 385.1910525
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910525, upper bound: 385.1910520
time: 8.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390
time: 7.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.16
Output dim: 1, lower bound: -385.1910520, upper bound: 385.1910525
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.16
Output dim: 1, lower bound: -385.1910525, upper bound: 385.1910520
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.16
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.16
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845778, upper bound: 385.1845768
time: 9.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845778, upper bound: 385.1845768
time: 9.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1196384, upper bound: 385.1196379
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1196384, upper bound: 385.1196379
time: 6.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1904210, upper bound: 385.1904210
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1904210, upper bound: 385.1904209
time: 9.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390
time: 8.07 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1845778, upper bound: 385.1845768
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1845778, upper bound: 385.1845768
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1196384, upper bound: 385.1196379
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1196384, upper bound: 385.1196379
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1904210, upper bound: 385.1904210
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1904210, upper bound: 385.1904209
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.05
Output dim: 1, lower bound: -385.1905390, upper bound: 385.1905390

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1749313, upper bound: 385.1749313
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1749313, upper bound: 385.1749313
time: 10.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845675, upper bound: 385.1845671
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845675, upper bound: 385.1845671
time: 10.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1899080, upper bound: 385.1899075
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1899080, upper bound: 385.1899075
time: 8.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256
time: 7.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1092284, upper bound: 385.1092187
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1092284, upper bound: 385.1092187
time: 7.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1904210, upper bound: 385.1904209
time: 9.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1904209, upper bound: 385.1904210
time: 9.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1749313, upper bound: 385.1749313
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1749313, upper bound: 385.1749313
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1845675, upper bound: 385.1845671
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1845675, upper bound: 385.1845671
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1899080, upper bound: 385.1899075
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1899080, upper bound: 385.1899075
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1092284, upper bound: 385.1092187
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1092284, upper bound: 385.1092187
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1904210, upper bound: 385.1904209
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.73
Output dim: 1, lower bound: -385.1904209, upper bound: 385.1904210

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1822703, upper bound: 385.1822708
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1822703, upper bound: 385.1822708
time: 9.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1842986, upper bound: 385.1842973
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1842986, upper bound: 385.1842973
time: 10.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1321485, upper bound: 385.1321529
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1321485, upper bound: 385.1321529
time: 6.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796383, upper bound: 385.1796297
time: 9.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796383, upper bound: 385.1796297
time: 8.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1879777, upper bound: 385.1879794
time: 9.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1879777, upper bound: 385.1879794
time: 10.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256
time: 8.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.31 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1822703, upper bound: 385.1822708
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1822703, upper bound: 385.1822708
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1842986, upper bound: 385.1842973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1842986, upper bound: 385.1842973
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1321485, upper bound: 385.1321529
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1321485, upper bound: 385.1321529
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1796383, upper bound: 385.1796297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1796383, upper bound: 385.1796297
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1879777, upper bound: 385.1879794
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1879777, upper bound: 385.1879794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.31
Output dim: 1, lower bound: -385.1756243, upper bound: 385.1756256

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1695614, upper bound: 385.1695551
time: 9.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1695614, upper bound: 385.1695551
time: 9.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1811866, upper bound: 385.1811934
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1811935, upper bound: 385.1811866
time: 9.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1710883, upper bound: 385.1710930
time: 9.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1710883, upper bound: 385.1710930
time: 8.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1830912, upper bound: 385.1831001
time: 10.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1831009, upper bound: 385.1830908
time: 13.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1772039, upper bound: 385.1771459
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1772039, upper bound: 385.1771459
time: 8.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1857971, upper bound: 385.1857942
time: 9.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1857971, upper bound: 385.1857942
time: 10.44 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.40 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1695614, upper bound: 385.1695551
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1695614, upper bound: 385.1695551
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1811866, upper bound: 385.1811934
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1811935, upper bound: 385.1811866
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1710883, upper bound: 385.1710930
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1710883, upper bound: 385.1710930
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1830912, upper bound: 385.1831001
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1831009, upper bound: 385.1830908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1772039, upper bound: 385.1771459
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1772039, upper bound: 385.1771459
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1857971, upper bound: 385.1857942
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.40
Output dim: 1, lower bound: -385.1857971, upper bound: 385.1857942

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1811866, upper bound: 385.1811931
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1811862, upper bound: 385.1811934
time: 9.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1801086, upper bound: 385.1800999
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1801086, upper bound: 385.1800999
time: 10.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1783179, upper bound: 385.1783371
time: 8.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1783179, upper bound: 385.1783371
time: 9.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1813943, upper bound: 385.1813480
time: 11.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1813943, upper bound: 385.1813480
time: 11.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1857490, upper bound: 385.1857942
time: 9.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1857971, upper bound: 385.1857476
time: 10.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1693055, upper bound: 385.1693060
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1693055, upper bound: 385.1693060
time: 8.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 17.80 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1811866, upper bound: 385.1811931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1811862, upper bound: 385.1811934
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1801086, upper bound: 385.1800999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1801086, upper bound: 385.1800999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1783179, upper bound: 385.1783371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1783179, upper bound: 385.1783371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1813943, upper bound: 385.1813480
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1813943, upper bound: 385.1813480
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1857490, upper bound: 385.1857942
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1857971, upper bound: 385.1857476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1693055, upper bound: 385.1693060
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 17.80
Output dim: 1, lower bound: -385.1693055, upper bound: 385.1693060
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=387.1527099609375
rel_dist={1: [-385.1965121761082, 385.19651217610806]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1936908, upper bound: 385.1936910
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1936910, upper bound: 385.1936908
time: 9.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.12
Output dim: 1, lower bound: -385.1936908, upper bound: 385.1936910
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.12
Output dim: 1, lower bound: -385.1936910, upper bound: 385.1936908

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1423897, upper bound: 385.1423895
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1423897, upper bound: 385.1423895
time: 8.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1936910, upper bound: 385.1936906
time: 9.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1936909, upper bound: 385.1936908
time: 9.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.85 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 19.85
Output dim: 1, lower bound: -385.1423897, upper bound: 385.1423895
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 19.85
Output dim: 1, lower bound: -385.1423897, upper bound: 385.1423895
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.85
Output dim: 1, lower bound: -385.1936910, upper bound: 385.1936906
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.85
Output dim: 1, lower bound: -385.1936909, upper bound: 385.1936908

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1928428, upper bound: 385.1928426
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1928428, upper bound: 385.1928426
time: 9.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1928339, upper bound: 385.1928314
time: 10.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1928339, upper bound: 385.1928314
time: 9.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.78 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 1, lower bound: -385.1928428, upper bound: 385.1928426
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 1, lower bound: -385.1928428, upper bound: 385.1928426
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 1, lower bound: -385.1928339, upper bound: 385.1928314
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.78
Output dim: 1, lower bound: -385.1928339, upper bound: 385.1928314

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1591501, upper bound: 385.1591427
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1591501, upper bound: 385.1591427
time: 8.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1915839, upper bound: 385.1915849
time: 9.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1915839, upper bound: 385.1915849
time: 12.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1911885, upper bound: 385.1911920
time: 12.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1911885, upper bound: 385.1911920
time: 11.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1928339, upper bound: 385.1928307
time: 11.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1928336, upper bound: 385.1928314
time: 10.13 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.33 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1591501, upper bound: 385.1591427
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1591501, upper bound: 385.1591427
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1915839, upper bound: 385.1915849
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1915839, upper bound: 385.1915849
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1911885, upper bound: 385.1911920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1911885, upper bound: 385.1911920
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1928339, upper bound: 385.1928307
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.33
Output dim: 1, lower bound: -385.1928336, upper bound: 385.1928314

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1307467, upper bound: 385.1307440
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1307467, upper bound: 385.1307440
time: 7.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1915627, upper bound: 385.1915849
time: 12.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1915839, upper bound: 385.1915659
time: 12.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910550, upper bound: 385.1910512
time: 10.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910493, upper bound: 385.1910562
time: 10.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1797646, upper bound: 385.1797672
time: 10.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1797646, upper bound: 385.1797672
time: 9.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845848, upper bound: 385.1845931
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845840, upper bound: 385.1845937
time: 10.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905143, upper bound: 385.1905245
time: 11.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905290, upper bound: 385.1905148
time: 10.27 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.54 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1307467, upper bound: 385.1307440
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1307467, upper bound: 385.1307440
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1915627, upper bound: 385.1915849
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1915839, upper bound: 385.1915659
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1910550, upper bound: 385.1910512
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1910493, upper bound: 385.1910562
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1797646, upper bound: 385.1797672
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1797646, upper bound: 385.1797672
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1845848, upper bound: 385.1845931
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1845840, upper bound: 385.1845937
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1905143, upper bound: 385.1905245
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.54
Output dim: 1, lower bound: -385.1905290, upper bound: 385.1905148

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1175496, upper bound: 385.1175382
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1175496, upper bound: 385.1175382
time: 7.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1175481, upper bound: 385.1175385
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1175481, upper bound: 385.1175385
time: 7.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1512086, upper bound: 385.1512115
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1512086, upper bound: 385.1512115
time: 7.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1835957, upper bound: 385.1835927
time: 10.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1835957, upper bound: 385.1835928
time: 11.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1842452, upper bound: 385.1842459
time: 12.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1842452, upper bound: 385.1842459
time: 11.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845571, upper bound: 385.1845650
time: 9.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845579, upper bound: 385.1845640
time: 10.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849950, upper bound: 385.1850030
time: 12.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849985, upper bound: 385.1849988
time: 8.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905290, upper bound: 385.1905139
time: 8.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1905277, upper bound: 385.1905148
time: 10.45 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 20.39 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1175496, upper bound: 385.1175382
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1175496, upper bound: 385.1175382
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1175481, upper bound: 385.1175385
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1175481, upper bound: 385.1175385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1512086, upper bound: 385.1512115
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1512086, upper bound: 385.1512115
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1835957, upper bound: 385.1835927
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1835957, upper bound: 385.1835928
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1842452, upper bound: 385.1842459
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1842452, upper bound: 385.1842459
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1845571, upper bound: 385.1845650
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1845579, upper bound: 385.1845640
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1849950, upper bound: 385.1850030
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1849985, upper bound: 385.1849988
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1905290, upper bound: 385.1905139
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 1, lower bound: -385.1905277, upper bound: 385.1905148

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1834673, upper bound: 385.1834611
time: 10.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1834673, upper bound: 385.1834611
time: 12.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1812300, upper bound: 385.1812300
time: 10.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1812300, upper bound: 385.1812300
time: 9.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1440086, upper bound: 385.1439981
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1440086, upper bound: 385.1439981
time: 7.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1800110, upper bound: 385.1800096
time: 9.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1800110, upper bound: 385.1800096
time: 10.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1740741, upper bound: 385.1740523
time: 9.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1740741, upper bound: 385.1740523
time: 9.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1755285, upper bound: 385.1755207
time: 9.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1755285, upper bound: 385.1755207
time: 10.49 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 20.94 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1834673, upper bound: 385.1834611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1834673, upper bound: 385.1834611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1812300, upper bound: 385.1812300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1812300, upper bound: 385.1812300
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1440086, upper bound: 385.1439981
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1440086, upper bound: 385.1439981
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1800110, upper bound: 385.1800096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1800110, upper bound: 385.1800096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1740741, upper bound: 385.1740523
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1740741, upper bound: 385.1740523
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1755285, upper bound: 385.1755207
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 20.94
Output dim: 1, lower bound: -385.1755285, upper bound: 385.1755207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.94
Output dim: 1, lower bound: -385.1849950, upper bound: 385.1850030
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.94
Output dim: 1, lower bound: -385.1849985, upper bound: 385.1849988
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.94
Output dim: 1, lower bound: -385.1905290, upper bound: 385.1905139
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.94
Output dim: 1, lower bound: -385.1905277, upper bound: 385.1905148
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=387.1527099609375
rel_dist={1: [-385.1964844738563, 385.19648447385634]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1941270, upper bound: 385.1941270
time: 10.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1941270, upper bound: 385.1941270
time: 11.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 22.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 22.31
Output dim: 1, lower bound: -385.1941270, upper bound: 385.1941270
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 22.31
Output dim: 1, lower bound: -385.1941270, upper bound: 385.1941270

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 244

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1941261, upper bound: 385.1941261
time: 17.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1941261, upper bound: 385.1941270
time: 10.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916571
time: 11.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916571
time: 11.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.71 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 1, lower bound: -385.1941261, upper bound: 385.1941261
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 1, lower bound: -385.1941261, upper bound: 385.1941270
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916571
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916571

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1872615, upper bound: 385.1872619
time: 15.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1872615, upper bound: 385.1872619
time: 12.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1941261, upper bound: 385.1941270
time: 10.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1941258, upper bound: 385.1941270
time: 12.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916565
time: 13.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916565, upper bound: 385.1916571
time: 12.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
time: 11.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916551, upper bound: 385.1916571
time: 14.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1872615, upper bound: 385.1872619
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1872615, upper bound: 385.1872619
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1941261, upper bound: 385.1941270
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1941258, upper bound: 385.1941270
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916565
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1916565, upper bound: 385.1916571
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 1, lower bound: -385.1916551, upper bound: 385.1916571

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1857339, upper bound: 385.1857339
time: 14.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1857339, upper bound: 385.1857339
time: 15.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1856613, upper bound: 385.1856613
time: 11.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1856613, upper bound: 385.1856613
time: 12.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1938756, upper bound: 385.1938762
time: 10.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1938756, upper bound: 385.1938762
time: 11.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1870602, upper bound: 385.1870611
time: 14.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1870602, upper bound: 385.1870611
time: 16.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1568692, upper bound: 385.1568717
time: 11.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1568692, upper bound: 385.1568717
time: 11.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1908831, upper bound: 385.1908831
time: 12.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1908831, upper bound: 385.1908845
time: 16.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 67

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
time: 11.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
time: 16.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916533, upper bound: 385.1916571
time: 10.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1916551, upper bound: 385.1916537
time: 17.23 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1857339, upper bound: 385.1857339
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1857339, upper bound: 385.1857339
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1856613, upper bound: 385.1856613
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1856613, upper bound: 385.1856613
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1938756, upper bound: 385.1938762
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1938756, upper bound: 385.1938762
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1870602, upper bound: 385.1870611
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1870602, upper bound: 385.1870611
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1568692, upper bound: 385.1568717
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1568692, upper bound: 385.1568717
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1908831, upper bound: 385.1908831
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1908831, upper bound: 385.1908845
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1916533, upper bound: 385.1916571
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.26
Output dim: 1, lower bound: -385.1916551, upper bound: 385.1916537

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1521601, upper bound: 385.1521594
time: 10.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1521601, upper bound: 385.1521594
time: 10.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1839390, upper bound: 385.1839390
time: 11.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1839390, upper bound: 385.1839390
time: 12.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1856598, upper bound: 385.1856613
time: 11.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1856598, upper bound: 385.1856598
time: 12.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1851548, upper bound: 385.1851548
time: 12.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1851548, upper bound: 385.1851548
time: 10.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1831915, upper bound: 385.1831930
time: 10.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1831915, upper bound: 385.1831930
time: 10.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910338, upper bound: 385.1910316
time: 12.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1910338, upper bound: 385.1910320
time: 12.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1801262, upper bound: 385.1801260
time: 10.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1801262, upper bound: 385.1801260
time: 11.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591
1: -210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100
2: -276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429
3: -294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862
4: -269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690
5: -241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797
6: -230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052
7: -251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947
8: -303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061
9: -228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1505502, upper bound: 385.1505610
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1505502, upper bound: 385.1505610
time: 7.95 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1521601, upper bound: 385.1521594
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1521601, upper bound: 385.1521594
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1839390, upper bound: 385.1839390
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1839390, upper bound: 385.1839390
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1856598, upper bound: 385.1856613
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1856598, upper bound: 385.1856598
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1851548, upper bound: 385.1851548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1851548, upper bound: 385.1851548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1831915, upper bound: 385.1831930
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1831915, upper bound: 385.1831930
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1910338, upper bound: 385.1910316
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1910338, upper bound: 385.1910320
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1801262, upper bound: 385.1801260
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1801262, upper bound: 385.1801260
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1505502, upper bound: 385.1505610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 17.11
Output dim: 1, lower bound: -385.1505502, upper bound: 385.1505610
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.11
Output dim: 1, lower bound: -385.1908831, upper bound: 385.1908831
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.11
Output dim: 1, lower bound: -385.1908831, upper bound: 385.1908845
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.11
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.11
Output dim: 1, lower bound: -385.1916571, upper bound: 385.1916551
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.11
Output dim: 1, lower bound: -385.1916533, upper bound: 385.1916571
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.11
Output dim: 1, lower bound: -385.1916551, upper bound: 385.1916537
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=387.1527099609375
rel_dist={1: [-385.1964274816625, 385.19642747694013]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1830.99 seconds
