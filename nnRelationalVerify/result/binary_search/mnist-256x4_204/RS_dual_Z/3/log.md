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
execution time: IAR + LP analysis = 1.10 + 10.83 = 11.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -385.1965521, upper bound: 385.1965521


# Binary Search by BASE starts (time budget: 2688.07 seconds, max iter: 100)

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
Binary search time: 43.57 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2644.50 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
time: 7.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.92
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.92
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849404, upper bound: 385.1849162
time: 9.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849162, upper bound: 385.1849404
time: 8.31 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849404, upper bound: 385.1849162
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849162, upper bound: 385.1849404
time: 8.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 1, lower bound: -385.1849404, upper bound: 385.1849162
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 1, lower bound: -385.1849162, upper bound: 385.1849404
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 1, lower bound: -385.1849404, upper bound: 385.1849162
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.26
Output dim: 1, lower bound: -385.1849162, upper bound: 385.1849404

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
time: 9.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
time: 9.67 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
time: 8.82 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
time: 10.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
time: 10.29 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
time: 8.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796354, upper bound: 385.1796128
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.28
Output dim: 1, lower bound: -385.1796128, upper bound: 385.1796354
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=387.1527099609375
rel_dist={1: [-385.1965121761082, 385.19651217610806]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850250, upper bound: 385.1850250
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850250, upper bound: 385.1850250
time: 8.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.68
Output dim: 1, lower bound: -385.1850250, upper bound: 385.1850250
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.68
Output dim: 1, lower bound: -385.1850250, upper bound: 385.1850250

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1849584
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849584, upper bound: 385.1849835
time: 8.49 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1849584
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849584, upper bound: 385.1849835
time: 8.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.59
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1849584
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.59
Output dim: 1, lower bound: -385.1849584, upper bound: 385.1849835
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.59
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1849584
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.59
Output dim: 1, lower bound: -385.1849584, upper bound: 385.1849835

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
time: 8.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
time: 8.21 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
time: 9.25 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
time: 8.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796655, upper bound: 385.1796392
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.57
Output dim: 1, lower bound: -385.1796392, upper bound: 385.1796654
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=387.1527099609375
rel_dist={1: [-385.1965333370153, 385.19653334436805]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850475, upper bound: 385.1850475
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850475, upper bound: 385.1850475
time: 9.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.19
Output dim: 1, lower bound: -385.1850475, upper bound: 385.1850475
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.19
Output dim: 1, lower bound: -385.1850475, upper bound: 385.1850475

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850070, upper bound: 385.1849835
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1850070
time: 8.75 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850070, upper bound: 385.1849835
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1850070
time: 8.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.30
Output dim: 1, lower bound: -385.1850070, upper bound: 385.1849835
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.30
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1850070
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.30
Output dim: 1, lower bound: -385.1850070, upper bound: 385.1849835
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.30
Output dim: 1, lower bound: -385.1849835, upper bound: 385.1850070

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
time: 7.52 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
time: 7.93 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
time: 9.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
time: 8.00 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
time: 7.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796835, upper bound: 385.1796556
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.65
Output dim: 1, lower bound: -385.1796556, upper bound: 385.1796835
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=387.1527099609375
rel_dist={1: [-385.1965462217105, 385.19654622393296]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850585, upper bound: 385.1850585
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850585, upper bound: 385.1850585
time: 7.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.37
Output dim: 1, lower bound: -385.1850585, upper bound: 385.1850585
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.37
Output dim: 1, lower bound: -385.1850585, upper bound: 385.1850585

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849940, upper bound: 385.1849940
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849940, upper bound: 385.1850184
time: 7.11 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850184, upper bound: 385.1849940
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849940, upper bound: 385.1850184
time: 6.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 1, lower bound: -385.1849940, upper bound: 385.1849940
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 1, lower bound: -385.1849940, upper bound: 385.1850184
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 1, lower bound: -385.1850184, upper bound: 385.1849940
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 1, lower bound: -385.1849940, upper bound: 385.1850184

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796632
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796924, upper bound: 385.1796632
time: 6.87 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
time: 7.45 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796924, upper bound: 385.1796632
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796924, upper bound: 385.1796632
time: 7.34 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
time: 7.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796632
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796924, upper bound: 385.1796632
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796924, upper bound: 385.1796632
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796924, upper bound: 385.1796632
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 15.21
Output dim: 1, lower bound: -385.1796632, upper bound: 385.1796924
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=387.1527099609375
rel_dist={1: [-385.1965520977046, 385.1965520977046]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 543.94 seconds
