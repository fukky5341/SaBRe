## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 154.56034074419998
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957)
1: (-70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644)
2: (-94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537)
3: (-99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775)
4: (-103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431)
5: (-81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278)
6: (-83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274)
7: (-88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160)
8: (-104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331)
9: (-84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249)

## BASE Result
execution time: IAR + LP analysis = 1.42 + 9.92 = 11.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7151429, upper bound: 154.7151428


# Binary Search by BASE starts (time budget: 1988.66 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=168.67294311523438
rel_dist={4: [-154.71512507779116, 154.715125091485]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=168.67294311523438
rel_dist={4: [-154.7150558205383, 154.7150558205383]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=168.67294311523438
rel_dist={4: [-154.71496529634885, 154.71496529634885]}

## Binary Search Result
Binary search time: 45.58 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1943.07 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6449324, upper bound: 154.6573072
time: 13.43 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6263141, upper bound: 154.6263141
time: 8.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.65 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 21.65
Output dim: 4, lower bound: -154.6449324, upper bound: 154.6573072
IS_B2, status: Status.UNKNOWN, split count: 1, time: 21.65
Output dim: 4, lower bound: -154.6263141, upper bound: 154.6263141

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -87.9151230, 68.5816803, -87.6555023, 68.3768311, -156.2919464, 156.2371826
1: -70.3884354, 62.2707710, -70.1716461, 62.0861206, -132.4745331, 132.4423981
2: -94.4465866, 64.4209595, -94.1626129, 64.2354660, -158.6820526, 158.5835724
3: -99.7921600, 55.1154213, -99.4909592, 54.9533005, -154.7454376, 154.6063690
4: -103.3198853, 65.3530502, -103.0291138, 65.1484680, -168.4683228, 168.3821716
5: -81.0754700, 65.9836578, -80.8302307, 65.7905731, -146.8660126, 146.8138885
6: -83.0478287, 77.3565063, -82.8069382, 77.1252670, -160.1730804, 160.1634216
7: -88.2885132, 75.0491028, -88.0284348, 74.8297195, -163.1182251, 163.0775452
8: -104.6999664, 72.3314667, -104.3845444, 72.1155777, -176.8155518, 176.7160034
9: -84.2536926, 75.4037476, -84.0119858, 75.1741028, -159.4277802, 159.4156952

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5507457, upper bound: 154.5457510
time: 13.37 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6425486, upper bound: 154.6549618
time: 11.38 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -87.6586838, 68.3795395, -99.6381149, 77.7250824, -165.3837433, 168.0175934
1: -70.1748428, 62.0885468, -79.7306213, 70.5418396, -140.7166748, 141.8190918
2: -94.1664352, 64.2375107, -107.1476135, 72.8780365, -167.0444641, 171.3850708
3: -99.4951553, 54.9549484, -113.0338058, 62.2619514, -161.7570648, 167.9887543
4: -103.0319443, 65.1520538, -117.3104477, 73.7910080, -176.8229065, 182.4624939
5: -80.8337708, 65.7930756, -91.8860626, 74.7644119, -155.5981598, 157.6791382
6: -82.8096924, 77.1283646, -94.1732941, 87.7177582, -170.5274506, 171.3016357
7: -88.0316620, 74.8323212, -100.0357056, 85.0350189, -173.0666809, 174.8680267
8: -104.3885803, 72.1180115, -118.5067368, 81.7048721, -186.0934448, 190.6247559
9: -84.0145950, 75.1777420, -95.5267792, 85.3593445, -169.3739319, 170.7045288

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5952868, upper bound: 154.5987836
time: 8.19 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5913017, upper bound: 154.5913017
time: 5.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.06 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 20.06
Output dim: 4, lower bound: -154.5507457, upper bound: 154.5457510
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 20.06
Output dim: 4, lower bound: -154.6425486, upper bound: 154.6549618
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 20.06
Output dim: 4, lower bound: -154.5952868, upper bound: 154.5987836
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 20.06
Output dim: 4, lower bound: -154.5913017, upper bound: 154.5913017

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -87.9151230, 68.5816803, -85.1796036, 66.4027100, -154.3178253, 153.7612762
1: -70.3884354, 62.2707710, -68.0809937, 60.3323860, -130.7208099, 130.3517456
2: -94.4465866, 64.4209595, -91.4297409, 62.4799004, -156.9264526, 155.8506775
3: -99.7921600, 55.1154213, -96.6078339, 53.4338226, -153.2259369, 151.7232513
4: -103.3198853, 65.3530502, -100.3919907, 63.0732574, -166.3931122, 165.7450409
5: -81.0754700, 65.9836578, -78.4552612, 63.9359894, -145.0114136, 144.4389191
6: -83.0478287, 77.3565063, -80.5355682, 74.9318237, -157.9796448, 157.8920746
7: -88.2885132, 75.0491028, -85.5535660, 72.7478790, -161.0363922, 160.6026459
8: -104.6999664, 72.3314667, -101.3887939, 70.0457001, -174.7456665, 173.7202454
9: -84.2536926, 75.4037476, -81.7520752, 72.9185867, -157.1722412, 157.1558228

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6268120, upper bound: 154.6406136
time: 12.09 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6121187, upper bound: 154.6290125
time: 13.49 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -87.6586838, 68.3795395, -88.1771774, 68.6310730, -156.2897644, 156.5567017
1: -70.1748428, 62.0885468, -70.1039734, 62.4541550, -132.6289978, 132.1924896
2: -94.1664352, 64.2375107, -94.5377045, 64.7451706, -158.9116058, 158.7751923
3: -99.4951553, 54.9549484, -99.7077942, 55.2300072, -154.7251282, 154.6627502
4: -103.0319443, 65.1520538, -105.1213379, 64.3628006, -167.3947449, 170.2733917
5: -80.8337708, 65.7930756, -80.9323502, 66.2396240, -147.0733948, 146.7254333
6: -82.8096924, 77.1283646, -83.6983719, 77.6105957, -160.4202881, 160.8267365
7: -88.0316620, 74.8323212, -88.6391068, 75.4566193, -163.4882812, 163.4714355
8: -104.3885803, 72.1180115, -104.6999588, 72.1882782, -176.5768433, 176.8179626
9: -84.0145950, 75.1777420, -85.0876770, 75.0355759, -159.0501556, 160.2653961

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912760, upper bound: 154.5936576
time: 10.16 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5737816, upper bound: 154.5785824
time: 7.25 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -85.3010712, 66.5167542, -90.8221741, 70.6583023, -155.9593658, 157.3389130
1: -68.1998901, 60.4231033, -72.1524200, 64.2817154, -132.4815979, 132.5755157
2: -91.5781250, 62.5599365, -97.3637238, 66.6522293, -158.2303467, 159.9236603
3: -96.7621994, 53.5062447, -102.6524048, 56.8709373, -153.6331329, 156.1586304
4: -100.4892197, 63.2387657, -108.4133377, 66.1444931, -166.6336975, 171.6520844
5: -78.5929871, 64.0393982, -83.3201828, 68.1759415, -146.7688904, 147.3595581
6: -80.6463394, 75.0465927, -86.2082520, 79.9323273, -160.5786743, 161.2548218
7: -85.6819153, 72.8550797, -91.2798157, 77.6821976, -163.3641052, 164.1348572
8: -101.5492477, 70.1687775, -107.8128662, 74.3655243, -175.9147644, 177.9816284
9: -81.8502960, 73.0633240, -87.6521988, 77.1712036, -159.0214996, 160.7155151

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5734176, upper bound: 154.5705765
time: 6.75 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5658193, upper bound: 154.5658197
time: 8.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.37 seconds
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 4, lower bound: -154.6268120, upper bound: 154.6406136
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 4, lower bound: -154.6121187, upper bound: 154.6290125
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 4, lower bound: -154.5912760, upper bound: 154.5936576
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 4, lower bound: -154.5737816, upper bound: 154.5785824
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 4, lower bound: -154.5734176, upper bound: 154.5705765
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 4, lower bound: -154.5658193, upper bound: 154.5658197

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: -76.1074295, 59.2191505, -85.1796036, 66.4027100, -142.5101318, 144.3987274
1: -60.4897194, 53.9429703, -68.0809937, 60.3323860, -120.8220901, 122.0239639
2: -81.4660339, 56.0410385, -91.4297409, 62.4799004, -143.9459076, 147.4707489
3: -86.0722733, 47.8770294, -96.6078339, 53.4338226, -139.5061035, 144.4848633
4: -90.7329102, 55.6714897, -100.3919907, 63.0732574, -153.8061523, 156.0634766
5: -69.8064270, 57.2018166, -78.4552612, 63.9359894, -133.7423859, 135.6570587
6: -72.2550507, 66.9468994, -80.5355682, 74.9318237, -147.1868744, 147.4824677
7: -76.5544891, 65.1790924, -85.5535660, 72.7478790, -149.3023682, 150.7326202
8: -90.4823608, 62.5359726, -101.3887939, 70.0457001, -160.5280609, 163.9247742
9: -73.4856491, 64.7898788, -81.7520752, 72.9185867, -146.4042053, 146.5419617

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B1_B2_A1_B1

### Relational analysis result of IS_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6222502, upper bound: 154.6358975
time: 12.91 seconds

## Relational analysis of IS_B1_B2_A1_B2

### Relational analysis result of IS_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6216976, upper bound: 154.6352643
time: 13.16 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -79.1272888, 61.5440216, -82.8502197, 64.5629196, -143.6902161, 144.3941956
1: -62.8410416, 56.0389481, -66.1300659, 58.6870041, -121.5280457, 122.1690140
2: -84.7055740, 58.2168427, -88.8725815, 60.8235016, -145.5290833, 147.0894165
3: -89.4422150, 49.7415771, -93.9058304, 52.0037766, -141.4459839, 143.6473846
4: -94.4524384, 57.7357559, -97.8819046, 61.1814957, -155.6339417, 155.6176605
5: -72.5466690, 59.4146614, -76.2410965, 62.2027740, -134.7494507, 135.6557617
6: -75.1119308, 69.6036377, -78.3975067, 72.8763504, -147.9882812, 148.0011292
7: -79.5699844, 67.7204742, -83.2330704, 70.7951660, -150.3651276, 150.9535522
8: -94.0417480, 65.0077591, -98.5839844, 68.1205368, -162.1622467, 163.5917358
9: -76.4039841, 67.2536469, -79.6152115, 70.8291702, -147.2331543, 146.8688507

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6030673, upper bound: 154.6180431
time: 12.26 seconds

## Relational analysis of IS_B1_B2_A2_B2

### Relational analysis result of IS_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6061275, upper bound: 154.6231321
time: 13.65 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -83.5724106, 65.1493759, -88.1771774, 68.6310730, -152.2034912, 153.3265533
1: -66.7672882, 59.1819458, -70.1039734, 62.4541550, -129.2214355, 129.2859039
2: -89.7004166, 61.3061714, -94.5377045, 64.7451706, -154.4455719, 155.8438568
3: -94.7246323, 52.4343643, -99.7077942, 55.2300072, -149.9546204, 152.1421509
4: -98.5418243, 61.8410454, -105.1213379, 64.3628006, -162.9046173, 166.9623718
5: -76.9485550, 62.6973038, -80.9323502, 66.2396240, -143.1881561, 143.6296539
6: -79.0190048, 73.5157089, -83.6983719, 77.6105957, -156.6295929, 157.2140808
7: -83.9175110, 71.3589249, -88.6391068, 75.4566193, -159.3741302, 159.9980316
8: -99.5225296, 68.7957230, -104.6999588, 72.1882782, -171.7108154, 173.4956818
9: -80.2292252, 71.5489197, -85.0876770, 75.0355759, -155.2647858, 156.6365662

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5774978, upper bound: 154.5809471
time: 7.83 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5847460, upper bound: 154.5872141
time: 9.16 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -91.5165710, 71.2473221, -87.0864868, 67.7689819, -159.2855530, 158.3338013
1: -73.0041428, 64.7931061, -69.1994095, 61.6780090, -134.6821442, 133.9925232
2: -98.1754456, 67.0829849, -93.3455276, 63.9606628, -162.1361084, 160.4284973
3: -103.5746307, 57.4036827, -98.4392853, 54.5563660, -158.1309967, 155.8429718
4: -108.3873138, 67.2430344, -103.9132004, 63.4865646, -171.8738708, 171.1562195
5: -84.0446548, 68.5454254, -79.8999557, 65.4114761, -149.4561157, 148.4453735
6: -86.6669540, 80.5153198, -82.6847382, 76.6468048, -163.3137512, 163.2000275
7: -91.8667755, 78.0837631, -87.5421143, 74.5299225, -166.3966980, 165.6258850
8: -108.9855118, 75.2783432, -103.4017715, 71.2984695, -180.2839813, 178.6800995
9: -87.9756088, 78.1058655, -84.0732193, 74.0719986, -162.0476074, 162.1790771

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5737816, upper bound: 154.5785824
time: 11.07 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5737816, upper bound: 154.5785824
time: 8.98 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -81.2174606, 63.2889786, -90.8221741, 70.6583023, -151.8757629, 154.1111450
1: -64.7946625, 57.5179863, -72.1524200, 64.2817154, -129.0763702, 129.6703949
2: -87.1150665, 59.6310883, -97.3637238, 66.6522293, -153.7673035, 156.9948120
3: -91.9925079, 50.9879227, -102.6524048, 56.8709373, -148.8634491, 153.6403198
4: -96.0014572, 59.9295807, -108.4133377, 66.1444931, -162.1459198, 168.3428802
5: -74.7108688, 60.9452591, -83.3201828, 68.1759415, -142.8867493, 144.2654419
6: -76.8569717, 71.4367599, -86.2082520, 79.9323273, -156.7893066, 157.6449738
7: -81.5698318, 69.3847733, -91.2798157, 77.6821976, -159.2520294, 160.6645813
8: -96.6866150, 66.8483505, -107.8128662, 74.3655243, -171.0521088, 174.6611633
9: -78.0672913, 69.4368515, -87.6521988, 77.1712036, -155.2384796, 157.0890503

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -154.5565105, upper bound: 154.5571532
time: 6.43 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5670245, upper bound: 154.5641531
time: 6.71 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -89.1759949, 69.3971252, -89.7244110, 69.7901382, -158.9661255, 159.1215363
1: -71.0409698, 63.1387634, -71.2421265, 63.5000000, -134.5409698, 134.3808899
2: -95.6042862, 65.4189453, -96.1637650, 65.8618469, -161.4661255, 161.5826721
3: -100.8587265, 55.9660149, -101.3756256, 56.1927490, -157.0514832, 157.3416443
4: -105.8718643, 65.3345337, -107.1952820, 65.2637024, -171.1355591, 172.5298157
5: -81.8175659, 66.8033447, -82.2814178, 67.3422470, -149.1597900, 149.0847626
6: -84.5207520, 78.4492950, -85.1879425, 78.9617615, -163.4825134, 163.6372375
7: -89.5335846, 76.1230774, -90.1750565, 76.7487564, -166.2823486, 166.2981262
8: -106.1682205, 73.3429871, -106.5054703, 73.4693680, -179.6375885, 179.8484497
9: -85.8320999, 76.0015411, -86.6302338, 76.2011337, -162.0332336, 162.6317749

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5658193, upper bound: 154.5658197
time: 5.79 seconds

## Relational analysis of IS_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5658193, upper bound: 154.5658197
time: 6.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 50.05 seconds
IS_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.6222502, upper bound: 154.6358975
IS_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.6216976, upper bound: 154.6352643
IS_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.6030673, upper bound: 154.6180431
IS_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.6061275, upper bound: 154.6231321
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5774978, upper bound: 154.5809471
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5847460, upper bound: 154.5872141
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5737816, upper bound: 154.5785824
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5737816, upper bound: 154.5785824
IS_B2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5565105, upper bound: 154.5571532
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5670245, upper bound: 154.5641531
IS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5658193, upper bound: 154.5658197
IS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 50.05
Output dim: 4, lower bound: -154.5658193, upper bound: 154.5658197

## BFS IS instance: IS_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -76.1074295, 59.2191505, -81.1001205, 63.1783295, -139.2857666, 140.3192749
1: -60.4897194, 53.9429703, -64.6800003, 57.4298134, -117.9195175, 118.6229706
2: -81.4660339, 56.0410385, -86.9712982, 59.5544472, -141.0204773, 143.0123138
3: -86.0722733, 47.8770294, -91.8444595, 50.9181976, -136.9904633, 139.7214813
4: -90.7329102, 55.6714897, -95.9076538, 59.7675323, -150.5004272, 151.5791473
5: -69.8064270, 57.2018166, -74.5777893, 60.8443565, -130.6507721, 131.7796021
6: -72.2550507, 66.9468994, -76.7496185, 71.3255234, -143.5805664, 143.6965179
7: -76.5544891, 65.1790924, -81.4456253, 69.2811737, -145.8356628, 146.6247253
8: -90.4823608, 62.5359726, -96.5308914, 66.7286224, -157.2109833, 159.0668640
9: -73.4856491, 64.7898788, -77.9727020, 69.2961426, -142.7817993, 142.7625732

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B2_A1_B1_B1

### Relational analysis result of IS_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6132692, upper bound: 154.6248729
time: 11.05 seconds

## Relational analysis of IS_B1_B2_A1_B1_B2

### Relational analysis result of IS_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6165439, upper bound: 154.6298292
time: 10.86 seconds

## BFS IS instance: IS_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -74.9888535, 58.3353577, -88.6797028, 68.9932861, -143.9821167, 147.0150452
1: -59.5619392, 53.1481056, -70.6416779, 62.7841034, -122.3460236, 123.7897797
2: -80.2441711, 55.2382736, -95.0620728, 65.0474777, -145.2916412, 150.3003540
3: -84.7710190, 47.1880341, -100.3101196, 55.6563225, -140.4273376, 147.4981384
4: -89.4971924, 54.7722054, -105.2485199, 64.9550858, -154.4522552, 160.0207062
5: -68.7472076, 56.3537827, -81.3657990, 66.4217682, -135.1689758, 137.7195740
6: -71.2178268, 65.9591599, -84.0376205, 78.0007858, -149.2185822, 149.9967804
7: -75.4302216, 64.2302246, -89.0258179, 75.6853638, -151.1155396, 153.2560425
8: -89.1526489, 61.6252556, -105.5471497, 72.9098053, -162.0624390, 167.1723785
9: -72.4477615, 63.8017731, -85.3394089, 75.5696106, -148.0173492, 149.1411743

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B2_A1_B2_B1

### Relational analysis result of IS_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6125316, upper bound: 154.6239994
time: 8.58 seconds

## Relational analysis of IS_B1_B2_A1_B2_B2

### Relational analysis result of IS_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6160709, upper bound: 154.6293475
time: 9.88 seconds

## BFS IS instance: IS_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -78.4641418, 61.0190506, -73.0708542, 56.8421555, -135.3063049, 134.0899048
1: -62.2815247, 55.5677185, -57.9131584, 51.7540092, -114.0355377, 113.4808807
2: -83.9791718, 57.7460556, -78.1913376, 53.8800735, -137.8592377, 135.9373932
3: -88.6749878, 49.3293304, -82.6120605, 45.9293823, -134.6043549, 131.9413910
4: -93.7305145, 57.1919365, -87.1860428, 53.2295990, -146.9600983, 144.3779755
5: -71.9181595, 58.9188423, -66.9998093, 54.9025574, -126.8206940, 125.9186554
6: -74.4949188, 69.0160294, -69.2976608, 64.2276230, -138.7225342, 138.3136902
7: -78.9045792, 67.1633759, -73.4469147, 62.5745239, -141.4790955, 140.6102600
8: -93.2339096, 64.4460297, -86.6997375, 59.8484688, -153.0823822, 151.1457520
9: -75.7926102, 66.6568756, -70.5925369, 62.0654144, -137.8580322, 137.2494202

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_B2_A2_B1_A1

### Relational analysis result of IS_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5972010, upper bound: 154.6136493
time: 10.18 seconds

## Relational analysis of IS_B1_B2_A2_B1_A2

### Relational analysis result of IS_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5972010, upper bound: 154.6180431
time: 16.09 seconds

## BFS IS instance: IS_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -78.8310394, 61.3100357, -77.4556656, 60.3043709, -139.1354065, 138.7656860
1: -62.5917587, 55.8286896, -61.5889091, 54.8637772, -117.4555359, 117.4176025
2: -84.3814850, 58.0067978, -82.9765930, 56.9990692, -141.3805542, 140.9833832
3: -89.0999222, 49.5574875, -87.6698914, 48.6517372, -137.7516632, 137.2273560
4: -94.1295853, 57.4937897, -92.0133972, 56.7742424, -150.9038239, 149.5071564
5: -72.2663498, 59.1933594, -71.1389465, 58.1739845, -130.4403381, 130.3322906
6: -74.8365173, 69.3413849, -73.3853989, 68.1047974, -142.9412842, 142.7267761
7: -79.2730179, 67.4717560, -77.8334427, 66.2643890, -145.5374146, 145.3052063
8: -93.6815109, 64.7575455, -92.0236282, 63.5638084, -157.2453156, 156.7811737
9: -76.1307373, 66.9877701, -74.6472015, 65.9813232, -142.1120605, 141.6349487

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_B2_A2_B2_A1

### Relational analysis result of IS_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5988833, upper bound: 154.6163704
time: 13.45 seconds

## Relational analysis of IS_B1_B2_A2_B2_A2

### Relational analysis result of IS_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5988833, upper bound: 154.6231321
time: 11.49 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -73.7087784, 57.3644180, -87.5365372, 68.1243515, -141.8330994, 144.9009399
1: -58.4793777, 52.1900291, -69.5629272, 61.9991074, -120.4784851, 121.7529449
2: -78.9263458, 54.3010712, -93.8359909, 64.2908859, -143.2172241, 148.1370544
3: -83.3352127, 46.3078613, -98.9663773, 54.8318634, -138.1670685, 145.2742310
4: -87.7372055, 53.8349228, -104.4255447, 63.8363457, -151.5735474, 158.2604523
5: -67.6303253, 55.3384476, -80.3248520, 65.7611694, -133.3914795, 135.6632996
6: -69.8398361, 64.7895203, -83.1025772, 77.0431747, -146.8830109, 147.8920898
7: -74.0441208, 63.0654907, -87.9970093, 74.9187851, -148.9629059, 151.0625000
8: -87.5340881, 60.4524689, -103.9195328, 71.6457214, -159.1797943, 164.3720093
9: -71.1219177, 62.7133484, -84.4983826, 74.4582977, -145.5802155, 147.2117310

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5768539, upper bound: 154.5797953
time: 8.73 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5768539, upper bound: 154.5809471
time: 8.59 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -78.1215668, 60.8459396, -87.8752899, 68.3927536, -146.5143127, 148.7212219
1: -62.1768570, 55.3182297, -69.8495636, 62.2400208, -124.4168777, 125.1677780
2: -83.7417526, 57.4408875, -94.2072830, 64.5314026, -148.2731476, 151.6481628
3: -88.4227600, 49.0466957, -99.3587875, 55.0425148, -143.4652557, 148.4054565
4: -92.6049728, 57.3929710, -104.7933655, 64.1154480, -156.7204285, 162.1863251
5: -71.7928772, 58.6283035, -80.6464844, 66.0142899, -137.8071442, 139.2747803
6: -73.9541473, 68.6913147, -83.4178543, 77.3433456, -151.2974854, 152.1091156
7: -78.4585037, 66.7799835, -88.3367920, 75.2031860, -153.6616821, 155.1167603
8: -92.8890610, 64.1904068, -104.3328018, 71.9334717, -164.8225250, 168.5232086
9: -75.2059555, 66.6506348, -84.8098831, 74.7641144, -149.9700623, 151.4605103

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5823300, upper bound: 154.5840243
time: 10.25 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5823300, upper bound: 154.5872141
time: 8.68 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -79.8897705, 62.0177345, -87.0864868, 67.7689819, -147.6587524, 149.1042175
1: -63.2420998, 56.5848694, -69.1994095, 61.6780090, -124.9201050, 125.7842636
2: -85.3780212, 58.8379478, -93.3455276, 63.9606628, -149.3386841, 152.1834564
3: -90.0537415, 50.2746010, -98.4392853, 54.5563660, -144.6100922, 148.7138824
4: -96.0230255, 57.6735420, -103.9132004, 63.4865646, -159.5095825, 161.5867157
5: -72.9305191, 59.8953667, -79.8999557, 65.4114761, -138.3419800, 139.7953186
6: -76.0457916, 70.2583466, -82.6847382, 76.6468048, -152.6925964, 152.9430695
7: -80.3054047, 68.3671341, -87.5421143, 74.5299225, -154.8353271, 155.9092255
8: -94.9797058, 65.6301880, -103.4017715, 71.2984695, -166.2781677, 169.0319519
9: -77.3867722, 67.6334305, -84.0732193, 74.0719986, -151.4587708, 151.7066498

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5650671, upper bound: 154.5686702
time: 11.32 seconds

## Relational analysis of IS_B2_B1_A2_A1_B2

### Relational analysis result of IS_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5672943, upper bound: 154.5721722
time: 7.28 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -82.4713898, 64.0043945, -87.0864868, 67.7689819, -150.2403412, 151.0908813
1: -65.2444763, 58.3705444, -69.1994095, 61.6780090, -126.9224701, 127.5699387
2: -88.1482697, 60.7008972, -93.3455276, 63.9606628, -152.1089325, 154.0464172
3: -92.9334183, 51.8692093, -98.4392853, 54.5563660, -147.4897766, 150.3084869
4: -99.2349854, 59.4145775, -103.9132004, 63.4865646, -162.7215576, 163.3277740
5: -75.2659912, 61.7835922, -79.8999557, 65.4114761, -140.6774597, 141.6835480
6: -78.4943390, 72.5301971, -82.6847382, 76.6468048, -155.1411438, 155.2149353
7: -82.8838959, 70.5410919, -87.5421143, 74.5299225, -157.4137726, 158.0832062
8: -98.0245972, 67.7566528, -103.4017715, 71.2984695, -169.3230591, 171.1584167
9: -79.8948059, 69.7220840, -84.0732193, 74.0719986, -153.9667969, 153.7953033

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 212

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5650671, upper bound: 154.5686698
time: 10.99 seconds

## Relational analysis of IS_B2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5672943, upper bound: 154.5721718
time: 9.91 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -75.7946930, 59.0078278, -90.5258484, 70.4242554, -146.2189484, 149.5336304
1: -60.2288437, 53.6757660, -71.9030304, 64.0714569, -124.3003006, 125.5787811
2: -81.1884918, 55.7874298, -97.0395660, 66.4420547, -147.6305389, 152.8269958
3: -85.7260742, 47.6184540, -102.3098907, 56.6868896, -142.4129333, 149.9283447
4: -90.0993958, 55.5015221, -108.0909424, 65.9023132, -156.0017090, 163.5924683
5: -69.5824432, 56.8961220, -83.0396957, 67.9547729, -137.5372162, 139.9358063
6: -71.8193970, 66.6387634, -85.9328156, 79.6700287, -151.4894104, 152.5715790
7: -76.1413345, 64.8307419, -90.9830170, 77.4334183, -153.5747528, 155.8137512
8: -90.0907898, 62.2674026, -107.4524536, 74.1152802, -164.2060699, 169.7198486
9: -73.0721512, 64.5651779, -87.3793945, 76.9048996, -149.9770355, 151.9445801

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=168.67294311523438
rel_dist={4: [-154.71512507779116, 154.715125091485]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6437439, upper bound: 154.6364807
time: 13.11 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6263043, upper bound: 154.6263043
time: 8.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.28
Output dim: 4, lower bound: -154.6437439, upper bound: 154.6364807
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.28
Output dim: 4, lower bound: -154.6263043, upper bound: 154.6263043

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -87.6555023, 68.3768311, -87.9151230, 68.5816803, -156.2371826, 156.2919464
1: -70.1716461, 62.0861206, -70.3884354, 62.2707710, -132.4423981, 132.4745178
2: -94.1626129, 64.2354660, -94.4465866, 64.4209595, -158.5835724, 158.6820374
3: -99.4909592, 54.9533005, -99.7921600, 55.1154213, -154.6063690, 154.7454376
4: -103.0291138, 65.1484680, -103.3198853, 65.3530502, -168.3821716, 168.4683228
5: -80.8302307, 65.7905731, -81.0754700, 65.9836578, -146.8138885, 146.8660126
6: -82.8069382, 77.1252670, -83.0478287, 77.3565063, -160.1634369, 160.1730652
7: -88.0284348, 74.8297195, -88.2885132, 75.0491028, -163.0775452, 163.1182251
8: -104.3845444, 72.1155777, -104.6999664, 72.3314667, -176.7160034, 176.8155518
9: -84.0119858, 75.1741028, -84.2536926, 75.4037476, -159.4156952, 159.4277802

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6240481, upper bound: 154.6164661
time: 11.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6131668, upper bound: 154.6035830
time: 11.19 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -99.6381149, 77.7250824, -87.2474747, 68.0554276, -167.6935120, 164.9725647
1: -79.7306213, 70.5418396, -69.8324356, 61.7963600, -141.5269318, 140.3742676
2: -107.1476135, 72.8780365, -93.7172241, 63.9433632, -171.0909424, 166.5952454
3: -113.0338058, 62.2619514, -99.0188751, 54.6976433, -167.7314453, 161.2808075
4: -117.3104477, 73.7910080, -102.5702438, 64.8297882, -182.1402283, 176.3612061
5: -91.8860626, 74.7644119, -80.4462433, 65.4875259, -157.3735962, 155.2106323
6: -94.1732941, 87.7177582, -82.4279099, 76.7625809, -170.9358521, 170.1456604
7: -100.0357056, 85.0350189, -87.6198730, 74.4847336, -174.5204315, 172.6548920
8: -118.5067368, 81.7048721, -103.8892441, 71.7757339, -190.2824402, 185.5941162
9: -95.5267792, 85.3593445, -83.6312256, 74.8153763, -170.3421478, 168.9905701

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5952179, upper bound: 154.5934074
time: 11.55 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 6.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.11 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 4, lower bound: -154.6240481, upper bound: 154.6164661
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 4, lower bound: -154.6131668, upper bound: 154.6035830
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 4, lower bound: -154.5952179, upper bound: 154.5934074
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 22.11
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -86.5459824, 67.4974747, -76.1074295, 59.2191505, -145.7651062, 143.6048889
1: -69.2417603, 61.3037567, -60.4897194, 53.9429703, -123.1847229, 121.7934723
2: -92.9429932, 63.4472427, -81.4660339, 56.0410385, -148.9840240, 144.9132690
3: -98.2028122, 54.2726326, -86.0722733, 47.8770294, -146.0798340, 140.3449097
4: -101.8439407, 64.2415543, -90.7329102, 55.6714897, -157.5154266, 154.9744568
5: -79.7717438, 64.9664383, -69.8064270, 57.2018166, -136.9735565, 134.7728577
6: -81.7932663, 76.1464920, -72.2550507, 66.9468994, -148.7401581, 148.4015503
7: -86.9255753, 73.9018707, -76.5544891, 65.1790924, -152.1046753, 150.4563599
8: -103.0481949, 71.1951675, -90.4823608, 62.5359726, -165.5841675, 161.6775208
9: -82.9993973, 74.1773071, -73.4856491, 64.7898788, -147.7892761, 147.6629639

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035170
time: 13.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035830
time: 13.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -81.8982849, 63.8293152, -79.1272888, 61.5440216, -143.4422913, 142.9566040
1: -65.3494415, 58.0205574, -62.8410416, 56.0389481, -121.3883896, 120.8615875
2: -87.8435440, 60.1394463, -84.7055740, 58.2168427, -146.0603638, 144.8450165
3: -92.8126221, 51.4170647, -89.4422150, 49.7415771, -142.5541534, 140.8592834
4: -96.8236694, 60.4723816, -94.4524384, 57.7357559, -154.5594177, 154.9247894
5: -75.3573151, 61.5067978, -72.5466690, 59.4146614, -134.7719727, 134.0534668
6: -77.5230713, 72.0439835, -75.1119308, 69.6036377, -147.1267090, 147.1559143
7: -82.2912064, 70.0015182, -79.5699844, 67.7204742, -150.0116882, 149.5714722
8: -97.4539261, 67.3574600, -94.0417480, 65.0077591, -162.4616699, 161.3991699
9: -78.7286072, 70.0092621, -76.4039841, 67.2536469, -145.9822388, 146.4132385

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035170
time: 11.23 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035830
time: 14.91 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -88.1771774, 68.6310730, -86.1373672, 67.1755447, -155.3527222, 154.7684326
1: -70.1039734, 62.4541550, -68.9019547, 61.0135574, -131.1175232, 131.3561096
2: -94.5377045, 64.7451706, -92.4969025, 63.1547737, -157.6924744, 157.2420654
3: -99.7077942, 55.2300072, -97.7299118, 54.0166359, -153.7244110, 152.9599152
4: -105.1213379, 64.3628006, -101.3844070, 63.9223747, -169.0436707, 165.7472076
5: -80.9323502, 66.2396240, -79.3871307, 64.6629562, -145.5953064, 145.6267548
6: -83.6983719, 77.6105957, -81.4136353, 75.7833099, -159.4816895, 159.0242310
7: -88.6391068, 75.4566193, -86.5163651, 73.5563660, -162.1954651, 161.9729919
8: -104.6999588, 72.1882782, -102.5521469, 70.8548050, -175.5547485, 174.7403870
9: -85.0876770, 75.0355759, -82.6181717, 73.8180313, -158.9056854, 157.6537476

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 7.07 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 6.95 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -90.8221741, 70.6583023, -81.5065689, 63.5204010, -154.3425751, 152.1648712
1: -72.1524200, 64.2817154, -65.0234756, 57.7420731, -129.8945007, 129.3051910
2: -97.3637238, 66.6522293, -87.4158554, 59.8589745, -157.2227020, 154.0680847
3: -102.6524048, 56.8709373, -92.3590393, 51.1712036, -153.8235931, 149.2299805
4: -108.4133377, 66.1444931, -96.3837128, 60.1655121, -168.5788574, 162.5281982
5: -83.3201828, 68.1759415, -74.9886246, 61.2153854, -144.5355682, 143.1645050
6: -86.2082520, 79.9323273, -77.1587524, 71.6960068, -157.9041901, 157.0910797
7: -91.2798157, 77.6821976, -81.8988495, 69.6702805, -160.9500732, 159.5810547
8: -107.8128662, 74.3655243, -96.9778290, 67.0306854, -174.8435364, 171.3433533
9: -87.6521988, 77.1712036, -78.3631058, 69.6643219, -157.3165283, 155.5343018

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 8.62 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 6.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.98 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035170
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035170
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035830
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 16.98
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -75.8630447, 59.0259247, -76.1074295, 59.2191505, -135.0821686, 135.1333618
1: -60.2853775, 53.7690048, -60.4897194, 53.9429703, -114.2283478, 114.2587128
2: -81.1988144, 55.8664894, -81.4660339, 56.0410385, -137.2398376, 137.3325195
3: -85.7888718, 47.7239532, -86.0722733, 47.8770294, -133.6658783, 133.7962341
4: -90.4597092, 55.4781036, -90.7329102, 55.6714897, -146.1311951, 146.2109833
5: -69.5755997, 57.0195541, -69.8064270, 57.2018166, -126.7774124, 126.8259811
6: -72.0279388, 66.7294312, -72.2550507, 66.9468994, -138.9748383, 138.9844818
7: -76.3097153, 64.9726639, -76.5544891, 65.1790924, -141.4888000, 141.5271606
8: -90.1852646, 62.3321877, -90.4823608, 62.5359726, -152.7212372, 152.8145447
9: -73.2582016, 64.5734177, -73.4856491, 64.7898788, -138.0480804, 138.0590668

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
time: 13.15 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
time: 11.80 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -78.8722000, 61.3423080, -76.1074295, 59.2191505, -138.0913239, 137.4497375
1: -62.6281967, 55.8573456, -60.4897194, 53.9429703, -116.5711670, 116.3470459
2: -84.4262772, 58.0346298, -81.4660339, 56.0410385, -140.4673157, 139.5006714
3: -89.1462402, 49.5819397, -86.0722733, 47.8770294, -137.0232544, 135.6542053
4: -94.1661987, 57.5342522, -90.7329102, 55.6714897, -149.8376617, 148.2671661
5: -72.3057098, 59.2242889, -69.8064270, 57.2018166, -129.5074921, 129.0307007
6: -74.8744659, 69.3764191, -72.2550507, 66.9468994, -141.8213501, 141.6314697
7: -79.3141174, 67.5049286, -76.5544891, 65.1790924, -144.4932098, 144.0594177
8: -93.7315140, 64.7954712, -90.4823608, 62.5359726, -156.2674866, 155.2778320
9: -76.1658707, 67.0278549, -73.4856491, 64.7898788, -140.9557495, 140.5135040

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
time: 14.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
time: 13.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -75.8630447, 59.0259247, -79.1272888, 61.5440216, -137.4070435, 138.1532135
1: -60.2853775, 53.7690048, -62.8410416, 56.0389481, -116.3243256, 116.6100464
2: -81.1988144, 55.8664894, -84.7055740, 58.2168427, -139.4156494, 140.5720673
3: -85.7888718, 47.7239532, -89.4422150, 49.7415771, -135.5304260, 137.1661682
4: -90.4597092, 55.4781036, -94.4524384, 57.7357559, -148.1954651, 149.9305115
5: -69.5755997, 57.0195541, -72.5466690, 59.4146614, -128.9902649, 129.5662231
6: -72.0279388, 66.7294312, -75.1119308, 69.6036377, -141.6315765, 141.8413544
7: -76.3097153, 64.9726639, -79.5699844, 67.7204742, -144.0301819, 144.5426331
8: -90.1852646, 62.3321877, -94.0417480, 65.0077591, -155.1930237, 156.3739319
9: -73.2582016, 64.5734177, -76.4039841, 67.2536469, -140.5118408, 140.9774017

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6009362, upper bound: 154.5936612
time: 12.72 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971538
time: 10.82 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -78.8722000, 61.3423080, -79.1272888, 61.5440216, -140.4161987, 140.4695892
1: -62.6281967, 55.8573456, -62.8410416, 56.0389481, -118.6671448, 118.6983643
2: -84.4262772, 58.0346298, -84.7055740, 58.2168427, -142.6431274, 142.7402039
3: -89.1462402, 49.5819397, -89.4422150, 49.7415771, -138.8878021, 139.0241547
4: -94.1661987, 57.5342522, -94.4524384, 57.7357559, -151.9019318, 151.9866791
5: -72.3057098, 59.2242889, -72.5466690, 59.4146614, -131.7203522, 131.7709656
6: -74.8744659, 69.3764191, -75.1119308, 69.6036377, -144.4781036, 144.4883423
7: -79.3141174, 67.5049286, -79.5699844, 67.7204742, -147.0345917, 147.0748749
8: -93.7315140, 64.7954712, -94.0417480, 65.0077591, -158.7392731, 158.8372040
9: -76.1658707, 67.0278549, -76.4039841, 67.2536469, -143.4195251, 143.4318390

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5985971, upper bound: 154.5893743
time: 11.39 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971874
time: 10.76 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -88.1771774, 68.6310730, -75.4747543, 58.7194176, -146.8965912, 144.1058044
1: -70.1039734, 62.4541550, -59.9623299, 53.4928551, -123.5968170, 122.4164734
2: -94.5377045, 64.7451706, -80.7750092, 55.5882759, -150.1259766, 145.5201721
3: -99.7077942, 55.2300072, -85.3398819, 47.4797287, -147.1875305, 140.5698853
4: -105.1213379, 64.3628006, -90.0229874, 55.1739922, -160.2953033, 154.3857880
5: -80.9323502, 66.2396240, -69.2102737, 56.7303314, -137.6626892, 135.4498901
6: -83.6983719, 77.6105957, -71.6662827, 66.3844757, -150.0828247, 149.2768555
7: -88.6391068, 75.4566193, -75.9207764, 64.6442795, -153.2833862, 151.3773956
8: -104.6999588, 72.1882782, -89.7133026, 62.0076180, -166.7075806, 161.9015808
9: -85.0876770, 75.0355759, -72.8953934, 64.2318954, -149.3195801, 147.9309692

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
time: 9.89 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
time: 9.54 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -88.1771774, 68.6310730, -78.4903641, 61.0406799, -149.2178650, 147.1214294
1: -70.1039734, 62.4541550, -62.3113670, 55.5858994, -125.6898346, 124.7655106
2: -94.5377045, 64.7451706, -84.0092316, 57.7605438, -152.2982330, 148.7543945
3: -99.7077942, 55.2300072, -88.7050476, 49.3415375, -149.0493317, 143.9350586
4: -105.1213379, 64.3628006, -93.7343063, 57.2366791, -162.3579712, 158.0971069
5: -80.9323502, 66.2396240, -71.9469604, 58.9394646, -139.8718109, 138.1865845
6: -83.6983719, 77.6105957, -74.5181122, 69.0370255, -152.7353973, 152.1286774
7: -88.6391068, 75.4566193, -78.9308853, 67.1816483, -155.8207550, 154.3875122
8: -104.6999588, 72.1882782, -93.2673187, 64.4765930, -169.1765442, 165.4555511
9: -85.0876770, 75.0355759, -75.8076401, 66.6926346, -151.7802734, 150.8432159

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
time: 8.86 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
time: 9.72 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -90.8221741, 70.6583023, -75.4747543, 58.7194176, -149.5415802, 146.1330261
1: -72.1524200, 64.2817154, -59.9623299, 53.4928551, -125.6452484, 124.2440414
2: -97.3637238, 66.6522293, -80.7750092, 55.5882759, -152.9519958, 147.4272156
3: -102.6524048, 56.8709373, -85.3398819, 47.4797287, -150.1321411, 142.2108154
4: -108.4133377, 66.1444931, -90.0229874, 55.1739922, -163.5873108, 156.1674805
5: -83.3201828, 68.1759415, -69.2102737, 56.7303314, -140.0505066, 137.3862152
6: -86.2082520, 79.9323273, -71.6662827, 66.3844757, -152.5926666, 151.5986023
7: -91.2798157, 77.6821976, -75.9207764, 64.6442795, -155.9240875, 153.6029663
8: -107.8128662, 74.3655243, -89.7133026, 62.0076180, -169.8204803, 164.0788269
9: -87.6521988, 77.1712036, -72.8953934, 64.2318954, -151.8840942, 150.0665588

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
time: 13.56 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
time: 9.04 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -90.8221741, 70.6583023, -78.4903641, 61.0406799, -151.8628387, 149.1486664
1: -72.1524200, 64.2817154, -62.3113670, 55.5858994, -127.7382812, 126.5930786
2: -97.3637238, 66.6522293, -84.0092316, 57.7605438, -155.1242676, 150.6614380
3: -102.6524048, 56.8709373, -88.7050476, 49.3415375, -151.9939270, 145.5759888
4: -108.4133377, 66.1444931, -93.7343063, 57.2366791, -165.6499786, 159.8787842
5: -83.3201828, 68.1759415, -71.9469604, 58.9394646, -142.2596436, 140.1228790
6: -86.2082520, 79.9323273, -74.5181122, 69.0370255, -155.2452393, 154.4504395
7: -91.2798157, 77.6821976, -78.9308853, 67.1816483, -158.4614563, 156.6130829
8: -107.8128662, 74.3655243, -93.2673187, 64.4765930, -172.2894287, 167.6327972
9: -87.6521988, 77.1712036, -75.8076401, 66.6926346, -154.3448334, 152.9788361

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
time: 7.99 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
time: 11.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.72 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.6009362, upper bound: 154.5936612
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971538
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5985971, upper bound: 154.5893743
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971874
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.72
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -72.2446747, 56.1646576, -66.5325394, 51.6589584, -123.9036331, 122.6971970
1: -57.2325706, 51.2009277, -52.4435539, 47.1499557, -104.3825226, 103.6444855
2: -77.2411880, 53.3007317, -71.0065384, 49.2442207, -126.4854126, 124.3072662
3: -81.6033325, 45.4756927, -75.0224991, 41.9286957, -123.5320206, 120.4981918
4: -86.5312576, 52.5105324, -80.2726822, 47.8808823, -134.4121246, 132.7832184
5: -66.1475143, 54.3139877, -60.7530212, 50.0444679, -116.1919708, 115.0670090
6: -68.6661682, 63.5286484, -63.3492393, 58.4781990, -127.1443634, 126.8778839
7: -72.6856384, 61.9359207, -66.9698410, 57.1305695, -129.8161926, 128.9057617
8: -85.7855530, 59.2655754, -78.8642807, 54.4325676, -140.2181244, 138.1298523
9: -69.9307327, 61.3171158, -64.6489258, 56.2133293, -126.1440582, 125.9660187

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6217867, upper bound: 154.6139410
time: 17.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6217867, upper bound: 154.6146722
time: 13.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -73.7258072, 57.3388710, -70.7142105, 54.9598579, -128.6856689, 128.0530701
1: -58.4858818, 52.2545967, -55.9476471, 50.1179848, -108.6038513, 108.2022324
2: -78.8644104, 54.3523026, -75.5692978, 52.2213478, -131.0857544, 129.9216003
3: -83.3191833, 46.3961258, -79.8393097, 44.5260658, -127.8452454, 126.2354202
4: -88.1393585, 53.7304688, -84.8793640, 51.2546043, -139.3939209, 138.6098022
5: -67.5536880, 55.4225388, -64.7002869, 53.1705017, -120.7241898, 120.1228180
6: -70.0442047, 64.8406754, -67.2483826, 62.1759148, -132.2201233, 132.0890350
7: -74.1715927, 63.1791687, -71.1540756, 60.6520348, -134.8236237, 134.3332520
8: -87.5901718, 60.5267143, -83.9303055, 57.9791565, -145.5693359, 144.4570160
9: -71.2929306, 62.6540565, -68.5222244, 59.9433098, -131.2362366, 131.1762848

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6219062, upper bound: 154.6142783
time: 11.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6219062, upper bound: 154.6160564
time: 12.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -75.2286224, 58.4575043, -66.5325394, 51.6589584, -126.8875580, 124.9900436
1: -59.5535393, 53.2683601, -52.4435539, 47.1499557, -106.7034912, 105.7119141
2: -80.4365158, 55.4481277, -71.0065384, 49.2442207, -129.6807404, 126.4546509
3: -84.9344635, 47.3154411, -75.0224991, 41.9286957, -126.8631516, 122.3379288
4: -90.2021332, 54.5470734, -80.2726822, 47.8808823, -138.0830078, 134.8197327
5: -68.8517532, 56.4999275, -60.7530212, 50.0444679, -118.8962250, 117.2529449
6: -71.4858246, 66.1466217, -63.3492393, 58.4781990, -129.9640045, 129.4958496
7: -75.6588745, 64.4440613, -66.9698410, 57.1305695, -132.7894135, 131.4139099
8: -89.2929688, 61.7100792, -78.8642807, 54.4325676, -143.7255096, 140.5743561
9: -72.8075104, 63.7494812, -64.6489258, 56.2133293, -129.0208282, 128.3984070

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6135560, upper bound: 154.6071832
time: 19.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6135560, upper bound: 154.6082560
time: 12.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -76.7928772, 59.6998024, -70.7142105, 54.9598579, -131.7527313, 130.4140015
1: -60.8784485, 54.3818855, -55.9476471, 50.1179848, -110.9964294, 110.3295288
2: -82.1516342, 56.5608215, -75.5692978, 52.2213478, -134.3729858, 132.1301270
3: -86.7446289, 48.2894974, -79.8393097, 44.5260658, -131.2706909, 128.1288147
4: -91.9015732, 55.8351898, -84.8793640, 51.2546043, -143.1561432, 140.7145386
5: -70.3378143, 57.6708488, -64.7002869, 53.1705017, -123.5083160, 122.3711319
6: -72.9420013, 67.5352631, -67.2483826, 62.1759148, -135.1179199, 134.7836456
7: -77.2300339, 65.7598343, -71.1540756, 60.6520348, -137.8820648, 136.9139099
8: -91.2024307, 63.0393600, -83.9303055, 57.9791565, -149.1815796, 146.9696655
9: -74.2485886, 65.1613770, -68.5222244, 59.9433098, -134.1918793, 133.6835938

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6136257, upper bound: 154.6074682
time: 18.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6136257, upper bound: 154.6098976
time: 11.41 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -66.2910919, 51.4681358, -75.4831696, 58.6588860, -124.9499588, 126.9512711
1: -52.2418594, 46.9781151, -59.7659492, 53.4495926, -105.6914444, 106.7440643
2: -70.7424850, 49.0714874, -80.7150345, 55.6300697, -126.3725281, 129.7865143
3: -74.7429428, 41.7774658, -85.2295532, 47.4747543, -122.2176819, 127.0070190
4: -80.0030212, 47.6896477, -90.4875793, 54.7480545, -134.7510681, 138.1772308
5: -60.5249977, 49.8640785, -69.0923004, 56.6899414, -117.2149353, 118.9563751
6: -63.1249008, 58.2633362, -71.7227249, 66.3734055, -129.4983063, 129.9860382
7: -66.7279816, 56.9266472, -75.9140930, 64.6590652, -131.3870544, 132.8407440
8: -78.5711823, 54.2308617, -89.6022797, 61.9219513, -140.4931335, 143.8331451
9: -64.4237823, 55.9998665, -73.0451202, 63.9746361, -128.3983917, 129.0449829

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5979257, upper bound: 154.5888262
time: 16.24 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5979257, upper bound: 154.5942628
time: 11.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 29.69 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6217867, upper bound: 154.6139410
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6217867, upper bound: 154.6146722
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6219062, upper bound: 154.6142783
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6219062, upper bound: 154.6160564
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6135560, upper bound: 154.6071832
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6135560, upper bound: 154.6082560
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6136257, upper bound: 154.6074682
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.6136257, upper bound: 154.6098976
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.5979257, upper bound: 154.5888262
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 29.69
Output dim: 4, lower bound: -154.5979257, upper bound: 154.5942628
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971538
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5985971, upper bound: 154.5893743
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971874
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.69
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=168.67294311523438
rel_dist={4: [-154.7150558205383, 154.7150558205383]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7114535, upper bound: 154.7112527
time: 11.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7110569, upper bound: 154.7110569
time: 10.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 22.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 22.92
Output dim: 4, lower bound: -154.7114535, upper bound: 154.7112527
IS_A2, status: Status.UNKNOWN, split count: 1, time: 22.92
Output dim: 4, lower bound: -154.7110569, upper bound: 154.7110569

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -84.9833069, 66.3094635, -85.7454453, 66.9006653, -151.8839722, 152.0549011
1: -68.0419388, 60.2050438, -68.6521378, 60.7422562, -128.7841949, 128.8571625
2: -91.2797928, 62.2895508, -92.1034622, 62.8435326, -154.1233063, 154.3930054
3: -96.4640884, 53.2908592, -97.3290482, 53.7655144, -150.2295990, 150.6199036
4: -99.8423157, 63.2022209, -100.7453461, 63.7628250, -163.6051331, 163.9475708
5: -78.3935013, 63.7917824, -79.0919647, 64.3618393, -142.7553406, 142.8837280
6: -80.2764435, 74.7869797, -80.9969788, 75.4553833, -155.7318268, 155.7839661
7: -85.3433609, 72.5609741, -86.1088791, 73.2079315, -158.5512695, 158.6698608
8: -101.2204132, 69.9331055, -102.1244354, 70.5571518, -171.7775269, 172.0575409
9: -81.4524384, 72.9249268, -82.1804581, 73.5700073, -155.0224457, 155.1053772

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
time: 13.50 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7019677, upper bound: 154.7016449
time: 14.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -93.6133347, 73.0096588, -85.6523438, 66.8295898, -160.4429169, 158.6620026
1: -75.0298996, 66.2731476, -68.5663071, 60.6755295, -135.7054138, 134.8394470
2: -100.5518494, 68.5826721, -91.9991837, 62.7806435, -163.3324890, 160.5818481
3: -106.2377243, 58.6899376, -97.2070847, 53.7049751, -159.9427032, 155.8969727
4: -109.9508972, 69.5278168, -100.6642685, 63.6812820, -173.6321411, 170.1920776
5: -86.2910233, 70.1964722, -79.0004272, 64.2942963, -150.5853119, 149.1968994
6: -88.3834763, 82.4106293, -80.9128342, 75.3744965, -163.7579346, 163.3234558
7: -94.0226898, 79.9081116, -86.0158691, 73.1331635, -167.1558533, 165.9239502
8: -111.5726776, 77.1049271, -102.0158234, 70.4873047, -182.0599365, 179.1207428
9: -89.7405777, 80.3339767, -82.1033630, 73.4868927, -163.2274780, 162.4373474

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
time: 12.60 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
time: 10.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.09 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.09
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.09
Output dim: 4, lower bound: -154.7019677, upper bound: 154.7016449
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 25.09
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 25.09
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -77.2854843, 60.2070427, -73.9434738, 57.5426598, -134.8281403, 134.1505127
1: -61.5895996, 54.7759705, -58.7595749, 52.4187050, -114.0083008, 113.5355377
2: -82.8180771, 56.8247375, -79.1312561, 54.4679832, -137.2860565, 135.9559784
3: -87.5196915, 48.5720291, -83.6173935, 46.5316772, -134.0513458, 132.1894226
4: -91.6275024, 56.8973427, -88.1627350, 54.0872192, -145.7146912, 145.0600739
5: -71.0473175, 58.0694275, -67.8279419, 55.5834694, -126.6307755, 125.8973618
6: -73.2404556, 67.9991074, -70.2107162, 65.0511475, -138.2915955, 138.2098236
7: -77.6925201, 66.1255951, -74.3811569, 63.3430939, -141.0356140, 140.5067444
8: -91.9499435, 63.5489845, -87.9176331, 60.7677994, -152.7177277, 151.4666138
9: -74.4293594, 66.0065231, -71.4166565, 62.9627533, -137.3921204, 137.4231720

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7022376, upper bound: 154.7019916
time: 10.94 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
time: 11.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -74.7869568, 58.2538185, -76.9468613, 59.8565559, -134.6434937, 135.2006683
1: -59.5046997, 53.0067024, -61.1012383, 54.5051346, -114.0098343, 114.1079330
2: -80.0929184, 55.0361023, -82.3547668, 56.6322975, -136.7252197, 137.3908691
3: -84.6377640, 47.0302048, -86.9714508, 48.3867950, -133.0245667, 134.0016479
4: -88.8565903, 54.9132195, -91.8559875, 56.1470680, -145.0036469, 146.7692108
5: -68.6991501, 56.1992912, -70.5555496, 57.7859802, -126.4851227, 126.7548141
6: -70.9158554, 65.7912827, -73.0507965, 67.6939774, -138.6098175, 138.8420715
7: -75.1854248, 64.0102463, -77.3781967, 65.8702545, -141.0556641, 141.3884430
8: -88.9557037, 61.5128021, -91.4599380, 63.2312622, -152.1869354, 152.9727325
9: -72.0960388, 63.7799911, -74.3162231, 65.4185257, -137.5145569, 138.0962219

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7019521, upper bound: 154.7016449
time: 15.39 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7019677, upper bound: 154.7016449
time: 11.04 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -81.8071442, 63.6465416, -77.9543228, 60.7267723, -142.5339203, 141.6008606
1: -65.1212769, 57.9445496, -62.1137772, 55.2462311, -120.3675003, 120.0583267
2: -87.5689316, 60.2088585, -83.5371933, 57.3160019, -144.8849182, 143.7460480
3: -92.5209122, 51.4492798, -88.2625427, 48.9854622, -141.5063629, 139.7118225
4: -97.3832932, 59.8357811, -92.4492645, 57.3752251, -154.7584839, 152.2850494
5: -75.0121918, 61.4231491, -71.6541519, 58.5713959, -133.5835876, 133.0772552
6: -77.6014786, 71.9935837, -73.8765564, 68.5867386, -146.1881866, 145.8701477
7: -82.2881317, 70.0382156, -78.3650208, 66.6978683, -148.9859619, 148.4032288
8: -97.3503876, 67.3095932, -92.7449417, 64.1023636, -161.4527588, 160.0545197
9: -78.9822311, 69.7060852, -75.0803070, 66.5681763, -145.5504150, 144.7863922

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
time: 12.74 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
time: 11.79 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -84.5745621, 65.7826004, -75.4350739, 58.7575111, -143.3320618, 141.2176819
1: -67.2848129, 59.8655090, -60.0125465, 53.4625587, -120.7473602, 119.8780518
2: -90.5446548, 62.1922455, -80.7898407, 55.5128860, -146.0575409, 142.9820862
3: -95.6154709, 53.1591644, -85.3577423, 47.4299622, -143.0454407, 138.5169067
4: -100.7577896, 61.7478523, -89.6536407, 55.3751602, -156.1329498, 151.4014740
5: -77.5357666, 63.4487762, -69.2873459, 56.6851578, -134.2209167, 132.7361145
6: -80.2140579, 74.4279251, -71.5321808, 66.3606415, -146.5747070, 145.9601135
7: -85.0446777, 72.3599548, -75.8369827, 64.5651398, -149.6098175, 148.1969299
8: -100.6119156, 69.5796661, -89.7259521, 62.0486717, -162.6605835, 159.3056030
9: -81.6378021, 71.9764023, -72.7276001, 64.3241959, -145.9620056, 144.7039795

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
time: 11.75 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
time: 10.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.88 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7022376, upper bound: 154.7019916
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7019521, upper bound: 154.7016449
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7019677, upper bound: 154.7016449
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -73.9683075, 57.6371956, -76.4457245, 59.6028824, -133.5711823, 134.0829010
1: -58.9111633, 52.4356880, -60.9278412, 54.1885033, -113.0996704, 113.3635254
2: -79.2304230, 54.4158440, -81.9504089, 56.1904144, -135.4208374, 136.3662262
3: -83.7675018, 46.5105019, -86.6623306, 48.0404968, -131.8079987, 133.1728363
4: -87.7154694, 54.4746742, -90.4615021, 56.4938354, -144.2093048, 144.9361420
5: -68.0107193, 55.6069756, -70.3708115, 57.4804993, -125.4912186, 125.9777756
6: -70.1299362, 65.0797501, -72.4422607, 67.2631760, -137.3931122, 137.5220032
7: -74.3655396, 63.3170586, -76.8445663, 65.4233627, -139.7889099, 140.1616211
8: -88.0013504, 60.8392639, -90.9609528, 62.8606377, -150.8619843, 151.8002014
9: -71.2670517, 63.1809235, -73.5721741, 65.3880768, -136.6551208, 136.7530975

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7022376, upper bound: 154.7019916
time: 11.28 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7022376, upper bound: 154.7019916
time: 10.89 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -76.6222534, 59.6934395, -72.9688492, 56.7875824, -133.4098358, 132.6622925
1: -61.0549164, 54.3081055, -57.9738922, 51.7313423, -112.7862549, 112.2819977
2: -82.1012573, 56.3446541, -78.0775070, 53.7619705, -135.8632202, 134.4221649
3: -86.7696381, 48.1603584, -82.5147400, 45.9267197, -132.6963501, 130.6750641
4: -90.8470001, 56.4131241, -87.0149384, 53.3762856, -144.2232819, 143.4280396
5: -70.4393692, 57.5776405, -66.9342957, 54.8605194, -125.2998886, 124.5119324
6: -72.6185226, 67.4152145, -69.2970810, 64.1930008, -136.8115234, 136.7122955
7: -77.0276413, 65.5645523, -73.4040604, 62.5182724, -139.5459137, 138.9685974
8: -91.1603470, 63.0081596, -86.7583923, 59.9735184, -151.1338501, 149.7665405
9: -73.7981873, 65.4428711, -70.4881821, 62.1346474, -135.9328156, 135.9310608

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
time: 11.21 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
time: 12.97 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -71.4972610, 55.7046242, -80.5692520, 62.7999001, -134.2971649, 136.2738800
1: -56.8483429, 50.6841965, -64.2142487, 57.0671387, -113.9154816, 114.8984375
2: -76.5345078, 52.6455765, -86.3962784, 59.1384010, -135.6729126, 139.0418549
3: -80.9149551, 44.9857330, -91.3225250, 50.5809097, -131.4958649, 136.3082581
4: -84.9715195, 52.5155640, -95.3497620, 59.4713287, -144.4428406, 147.8653107
5: -65.6874619, 53.7565804, -74.1661530, 60.5133781, -126.2008362, 127.9227295
6: -67.8299103, 62.8953819, -76.3076935, 70.8963928, -138.7262878, 139.2030640
7: -71.8849335, 61.2225990, -80.9602661, 68.8838196, -140.7687531, 142.1828156
8: -85.0419617, 58.8251877, -95.8522186, 66.2516098, -151.2935638, 154.6773834
9: -68.9556351, 60.9786034, -77.4980621, 68.8525238, -137.8081665, 138.4766693

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6882219, upper bound: 154.6873300
time: 13.65 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6871638, upper bound: 154.6867881
time: 10.37 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -74.1368866, 57.7501602, -75.9479752, 59.0829086, -133.2197876, 133.6981354
1: -58.9809570, 52.5482903, -60.2968140, 53.8017197, -112.7826767, 112.8451080
2: -79.3900223, 54.5650444, -81.2758636, 55.9070778, -135.2971039, 135.8408966
3: -83.9023743, 46.6265030, -85.8422928, 47.7660027, -131.6683807, 132.4687958
4: -88.0887680, 54.4402809, -90.6748352, 55.4219971, -143.5107574, 145.1150970
5: -68.1036453, 55.7170143, -69.6412964, 57.0447655, -125.1483917, 125.3583069
6: -70.3058853, 65.2188873, -72.1127930, 66.8152924, -137.1211700, 137.3316650
7: -74.5333862, 63.4596901, -76.3773117, 65.0237808, -139.5571442, 139.8369751
8: -88.1823959, 60.9829521, -90.2719193, 62.4159622, -150.5983582, 151.2548370
9: -71.4757385, 63.2280617, -73.3627014, 64.5711975, -136.0469360, 136.5907593

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6890637, upper bound: 154.6892921
time: 14.44 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6886205, upper bound: 154.6881593
time: 11.49 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -81.8071442, 63.6465416, -73.8472977, 57.4687881, -139.2759399, 137.4938354
1: -65.1212769, 57.9445496, -58.6715050, 52.3498497, -117.4711227, 116.6160583
2: -87.5689316, 60.2088585, -79.0241852, 54.4033241, -141.9722443, 139.2330322
3: -92.5209122, 51.4492798, -83.4925461, 46.4691086, -138.9899902, 134.9418335
4: -97.3832932, 59.8357811, -88.0774231, 54.0018387, -151.3850708, 147.9132080
5: -75.0121918, 61.4231491, -67.7338943, 55.5130539, -130.5252380, 129.1570435
6: -77.6014786, 71.9935837, -70.1233368, 64.9680939, -142.5695190, 142.1169128
7: -82.2881317, 70.0382156, -74.2854462, 63.2661743, -145.5542908, 144.3236694
8: -97.3503876, 67.3095932, -87.8053513, 60.6947823, -158.0451660, 155.1149139
9: -78.9822311, 69.7060852, -71.3372116, 62.8769569, -141.8591614, 141.0433044

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7014864, upper bound: 154.7014071
time: 12.08 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
time: 12.22 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -81.8071442, 63.6465416, -76.8297043, 59.7660828, -141.5732269, 140.4762421
1: -65.1212769, 57.9445496, -60.9970703, 54.4210663, -119.5423431, 118.9416046
2: -87.5689316, 60.2088585, -82.2249985, 56.5511589, -144.1200867, 142.4338531
3: -92.5209122, 51.4492798, -86.8236618, 48.3102646, -140.8311768, 138.2729034
4: -97.3832932, 59.8357811, -91.7391052, 56.0455208, -153.4287872, 151.5748749
5: -75.0121918, 61.4231491, -70.4431000, 57.6981354, -132.7103271, 131.8662415
6: -77.6014786, 71.9935837, -72.9417419, 67.5918121, -145.1932831, 144.9353333
7: -82.2881317, 70.0382156, -77.2603607, 65.7744751, -148.0625763, 147.2985687
8: -97.3503876, 67.3095932, -91.3229141, 63.1401482, -160.4905243, 158.6325073
9: -78.9822311, 69.7060852, -74.2140884, 65.3152237, -144.2974548, 143.9201660

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7014864, upper bound: 154.7014071
time: 12.05 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
time: 13.35 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -81.2107544, 63.1768341, -78.9731598, 61.6305885, -142.8413239, 142.1499634
1: -64.5676880, 57.4937057, -63.0481720, 55.9646492, -120.5323257, 120.5418777
2: -86.9068756, 59.7460365, -84.7360458, 57.9701462, -144.8769989, 144.4820251
3: -91.8112488, 51.0658302, -89.6001129, 49.5756226, -141.3868408, 140.6659088
4: -96.7855682, 59.2943039, -93.0870056, 58.6023102, -155.3878784, 152.3812714
5: -74.4580154, 60.9502563, -72.8074570, 59.3490067, -133.8070221, 133.7577209
6: -77.0566101, 71.4674454, -74.7153015, 69.4890289, -146.5456390, 146.1827240
7: -81.6714859, 69.5103989, -79.3314133, 67.5129776, -149.1844635, 148.8417969
8: -96.6072998, 66.8274689, -94.0186844, 64.9975967, -161.6048737, 160.8461304
9: -78.4268646, 69.1152878, -75.8423767, 67.6737213, -146.1005859, 144.9576111

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
time: 11.65 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
time: 11.28 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -83.9032745, 65.2626114, -74.4705200, 58.0100670, -141.9133148, 139.7331238
1: -66.7440872, 59.3927879, -59.2350845, 52.7823601, -119.5264435, 118.6278687
2: -89.8191833, 61.7049561, -79.7467346, 54.8134079, -144.6325836, 141.4516907
3: -94.8567276, 52.7416191, -84.2664566, 46.8308563, -141.6875763, 137.0080719
4: -99.9640732, 61.2591553, -88.5153961, 54.6720467, -154.6361084, 149.7745209
5: -76.9212875, 62.9505234, -68.4032135, 55.9692650, -132.8905487, 131.3537292
6: -79.5835190, 73.8370743, -70.6274948, 65.5112381, -145.0947571, 144.4645691
7: -84.3718643, 71.7912216, -74.8695908, 63.7483444, -148.1201935, 146.6607971
8: -99.8127289, 69.0312653, -88.5785675, 61.2623672, -161.0751038, 157.6098328
9: -80.9968643, 71.4067459, -71.8076859, 63.5047913, -144.5016479, 143.2144318

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6883832, upper bound: 154.6890810
time: 11.76 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6878031, upper bound: 154.6878034
time: 10.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.87 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7022376, upper bound: 154.7019916
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7022376, upper bound: 154.7019916
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7022408, upper bound: 154.7019916
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.6882219, upper bound: 154.6873300
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.6871638, upper bound: 154.6867881
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.6890637, upper bound: 154.6892921
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.6886205, upper bound: 154.6881593
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7014864, upper bound: 154.7014071
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7014864, upper bound: 154.7014071
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.6883832, upper bound: 154.6890810
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 4, lower bound: -154.6878031, upper bound: 154.6878034

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -69.8964005, 54.4061623, -76.4457245, 59.6028824, -129.4992828, 130.8518677
1: -55.4974861, 49.5622826, -60.9278412, 54.1885033, -109.6859894, 110.4901199
2: -74.7560654, 51.5259094, -81.9504089, 56.1904144, -130.9464722, 133.4762878
3: -79.0355606, 44.0170326, -86.6623306, 48.0404968, -127.0760574, 130.6793671
4: -83.3806305, 51.1341820, -90.4615021, 56.4938354, -139.8744659, 141.5956573
5: -64.1219711, 52.5731087, -70.3708115, 57.4804993, -121.6024704, 122.9439240
6: -66.4088898, 61.4920311, -72.4422607, 67.2631760, -133.6720581, 133.9342957
7: -70.3212128, 59.9123688, -76.8445663, 65.4233627, -135.7445679, 136.7569275
8: -83.1102219, 57.4611320, -90.9609528, 62.8606377, -145.9708557, 148.4220886
9: -67.5519791, 59.5206223, -73.5721741, 65.3880768, -132.9400635, 133.0927734

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6899176, upper bound: 154.6892715
time: 13.35 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6880150, upper bound: 154.6879766
time: 12.52 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -72.8211823, 56.6610222, -76.4457245, 59.6028824, -132.4240723, 133.1067352
1: -57.7787514, 51.5960541, -60.9278412, 54.1885033, -111.9672546, 112.5238953
2: -77.8984299, 53.6289101, -81.9504089, 56.1904144, -134.0888367, 135.5793152
3: -82.3048019, 45.8224983, -86.6623306, 48.0404968, -130.3452759, 132.4848328
4: -86.9709396, 53.1494255, -90.4615021, 56.4938354, -143.4647827, 143.6109314
5: -66.7826996, 54.7171555, -70.3708115, 57.4804993, -124.2631989, 125.0879669
6: -69.1716156, 64.0679169, -72.4422607, 67.2631760, -136.4347687, 136.5101776
7: -73.2410583, 62.3717575, -76.8445663, 65.4233627, -138.6644287, 139.2163239
8: -86.5609589, 59.8591995, -90.9609528, 62.8606377, -149.4215851, 150.8201599
9: -70.3728333, 61.9171791, -73.5721741, 65.3880768, -135.7609100, 135.4893494

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6899176, upper bound: 154.6892708
time: 13.66 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6880150, upper bound: 154.6879767
time: 12.49 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -72.5251007, 56.4430542, -72.9688492, 56.7875824, -129.3126678, 129.4118958
1: -57.6210861, 51.4188690, -57.9738922, 51.7313423, -109.3524094, 109.3927460
2: -77.5985870, 53.4380646, -78.0775070, 53.7619705, -131.3605652, 131.5155640
3: -82.0104370, 45.6504173, -82.5147400, 45.9267197, -127.9371567, 128.1651154
4: -86.4843597, 53.0495110, -87.0149384, 53.3762856, -139.8606415, 140.0644379
5: -66.5284348, 54.5267906, -66.9342957, 54.8605194, -121.3889389, 121.4610672
6: -68.8748245, 63.8052864, -69.2970810, 64.1930008, -133.0678253, 133.1023712
7: -72.9577866, 62.1407242, -73.4040604, 62.5182724, -135.4760590, 135.5447693
8: -86.2340012, 59.6092377, -86.7583923, 59.9735184, -146.2075195, 146.3676300
9: -70.0624771, 61.7608490, -70.4881821, 62.1346474, -132.1970978, 132.2489929

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6908293, upper bound: 154.6900174
time: 13.59 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6880146, upper bound: 154.6889226
time: 12.66 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -75.4424210, 58.6920090, -72.9688492, 56.7875824, -132.2299957, 131.6608582
1: -59.8982697, 53.4467392, -57.9738922, 51.7313423, -111.6296082, 111.4206314
2: -80.7321930, 55.5367737, -78.0775070, 53.7619705, -134.4941711, 133.6142731
3: -85.2709732, 47.4514809, -82.5147400, 45.9267197, -131.1976929, 129.9662018
4: -90.0610123, 55.0600891, -87.0149384, 53.3762856, -143.4373016, 142.0750122
5: -69.1828384, 56.6655312, -66.9342957, 54.8605194, -124.0433578, 123.5998230
6: -71.6296387, 66.3738403, -69.2970810, 64.1930008, -135.8226166, 135.6709290
7: -75.8680496, 64.5926361, -73.4040604, 62.5182724, -138.3863220, 137.9966888
8: -89.6754379, 62.0019836, -86.7583923, 59.9735184, -149.6489563, 148.7603607
9: -72.8735352, 64.1505203, -70.4881821, 62.1346474, -135.0081787, 134.6387024

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6908293, upper bound: 154.6900166
time: 11.66 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6880146, upper bound: 154.6889230
time: 11.78 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -69.4377670, 54.1022568, -79.0601044, 61.6273041, -131.0650635, 133.1623535
1: -55.1783752, 49.2160072, -62.9940758, 55.9956665, -111.1740417, 112.2100677
2: -74.3079834, 51.1401138, -84.7676620, 58.0387344, -132.3466949, 135.9077606
3: -78.5639114, 43.6970482, -89.6025772, 49.6407700, -128.2046814, 133.2995911
4: -82.5207825, 50.9999695, -93.5646896, 58.3604622, -140.8812256, 144.5646667
5: -63.7883606, 52.2059593, -72.7754898, 59.3800087, -123.1683655, 124.9814377
6: -65.8775711, 61.0692253, -74.8825912, 69.5623856, -135.4399567, 135.9517975
7: -69.8075714, 59.4540329, -79.4426117, 67.5914154, -137.3989563, 138.8966370
8: -82.5765152, 57.1288490, -94.0507126, 65.0141525, -147.5906677, 151.1795654
9: -66.9603424, 59.2234993, -76.0413132, 67.5686874, -134.5290222, 135.2648010

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6782060, upper bound: 154.6771252
time: 12.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6779525, upper bound: 154.6767436
time: 12.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -80.5240860, 62.8456001, -77.9745941, 60.7838783, -141.3079529, 140.8201904
1: -64.5505447, 57.1005516, -62.1164665, 55.2290649, -119.7796097, 119.2170181
2: -86.4963913, 58.9618187, -83.5963135, 57.2577477, -143.7541199, 142.5581055
3: -91.6288300, 50.6480217, -88.3605652, 48.9707603, -140.5995941, 139.0085754
4: -94.3654709, 60.0181961, -92.3138580, 57.5387917, -151.9042664, 152.3320312
5: -74.3164368, 60.4191170, -71.7692566, 58.5663490, -132.8827820, 132.1883545
6: -75.9655075, 70.9349213, -73.8678894, 68.6086121, -144.5741272, 144.8028107
7: -80.8196716, 68.6983490, -78.3612900, 66.6655884, -147.4852600, 147.0596313
8: -95.8413467, 66.2798004, -92.7537689, 64.1226044, -159.9639587, 159.0335693
9: -77.0292969, 69.0220566, -75.0046997, 66.6354752, -143.6647644, 144.0267639

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6871638, upper bound: 154.6867881
time: 11.02 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6871638, upper bound: 154.6867877
time: 10.56 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -72.6764832, 56.6143608, -73.7636261, 57.3860092, -130.0624542, 130.3779907
1: -57.7979431, 51.5088158, -58.5310364, 52.2510147, -110.0489578, 110.0398560
2: -77.8128433, 53.4998894, -78.9193573, 54.3157196, -132.1285706, 132.4192505
3: -82.2372437, 45.7141418, -83.3537521, 46.4035492, -128.6407776, 129.0678711
4: -86.3543320, 53.3664818, -88.0940552, 53.8134766, -140.1678162, 141.4605408
5: -66.7579422, 54.6192245, -67.6279221, 55.4034920, -122.1614380, 122.2471313
6: -68.9230042, 63.9258347, -70.0505676, 64.8853149, -133.8082886, 133.9763947
7: -73.0619888, 62.2073975, -74.1811523, 63.1539764, -136.2159576, 136.3885345
8: -86.4350967, 59.7827263, -87.6656113, 60.6221085, -147.0572052, 147.4483337
9: -70.0638275, 61.9847832, -71.2541962, 62.7135239, -132.7773438, 133.2389679

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6890641, upper bound: 154.6892921
time: 12.40 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6890641, upper bound: 154.6892921
time: 14.15 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -71.5264282, 55.7202415, -84.9905701, 66.2434998, -137.7699127, 140.7107697
1: -56.8681488, 50.6933403, -68.0191498, 60.2325745, -117.1007233, 118.7124939
2: -76.5704041, 52.6688271, -91.2739716, 62.2266731, -138.7970734, 143.9427948
3: -80.9226151, 44.9985161, -96.5828171, 53.4418869, -134.3645020, 141.5812988
4: -85.0104141, 52.5033607, -100.1003113, 62.9698181, -147.9802094, 152.6036682
5: -65.6955414, 53.7542534, -78.2881851, 63.7216339, -129.4171753, 132.0424347
6: -67.8412247, 62.9119034, -80.2695160, 74.8795853, -142.7208099, 143.1814270
7: -71.9121170, 61.2231827, -85.3448868, 72.5123672, -144.4244843, 146.5680542
8: -85.0570068, 58.8347473, -101.1254044, 69.8912354, -154.9482422, 159.9601440
9: -68.9585800, 60.9978104, -81.4481583, 72.6334381, -141.5920105, 142.4459686

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 123

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6802524, upper bound: 154.6795070
time: 10.31 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6847305, upper bound: 154.6842237
time: 12.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.51 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6899176, upper bound: 154.6892715
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6880150, upper bound: 154.6879766
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6899176, upper bound: 154.6892708
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6880150, upper bound: 154.6879767
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6908293, upper bound: 154.6900174
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6880146, upper bound: 154.6889226
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6908293, upper bound: 154.6900166
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6880146, upper bound: 154.6889230
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6782060, upper bound: 154.6771252
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6779525, upper bound: 154.6767436
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6871638, upper bound: 154.6867881
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6871638, upper bound: 154.6867877
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6890641, upper bound: 154.6892921
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6890641, upper bound: 154.6892921
IS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6802524, upper bound: 154.6795070
IS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.51
Output dim: 4, lower bound: -154.6847305, upper bound: 154.6842237
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.7014864, upper bound: 154.7014071
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.7014864, upper bound: 154.7014071
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.7014967, upper bound: 154.7014071
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.7012822, upper bound: 154.7012822
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.6883832, upper bound: 154.6890810
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.51
Output dim: 4, lower bound: -154.6878031, upper bound: 154.6878034
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=168.67294311523438
rel_dist={4: [-154.71496529634885, 154.71496529634885]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1834.29 seconds
