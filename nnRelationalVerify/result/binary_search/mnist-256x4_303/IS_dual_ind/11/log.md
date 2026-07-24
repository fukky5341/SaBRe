## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 330.610742861
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-183.6371460, 146.0147095, -183.6371460, 146.0147095, -329.6518250, 329.6518250)
1: (-154.1186218, 129.7214966, -154.1186218, 129.7214966, -283.8400879, 283.8400879)
2: (-202.5823517, 131.3443756, -202.5823517, 131.3443756, -333.9267273, 333.9267273)
3: (-215.1947327, 113.4665604, -215.1947327, 113.4665604, -328.6612854, 328.6612854)
4: (-197.8540955, 151.6625977, -197.8540955, 151.6625977, -349.5166931, 349.5166931)
5: (-176.9648895, 137.6301270, -176.9648895, 137.6301270, -314.5950317, 314.5950317)
6: (-169.7169952, 162.8930969, -169.7169952, 162.8930969, -332.6100464, 332.6100464)
7: (-184.4264832, 154.7376404, -184.4264832, 154.7376404, -339.1640930, 339.1640930)
8: (-221.7955170, 151.2197571, -221.7955170, 151.2197571, -373.0152588, 373.0152588)
9: (-167.8070984, 165.4597626, -167.8070984, 165.4597626, -333.2668152, 333.2668152)

## BASE Result
execution time: IAR + LP analysis = 1.23 + 10.49 = 11.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -330.6222845, upper bound: 330.6222845


# Binary Search by BASE starts (time budget: 2688.28 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=333.2668151855469
rel_dist={9: [-330.6221776084849, 330.6221776084849]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=333.2668151855469
rel_dist={9: [-330.62201561424183, 330.62201560609844]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=333.2668151855469
rel_dist={9: [-330.62178962803773, 330.6217896201217]}

## Binary Search Result
Binary search time: 48.25 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2640.03 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6216385, upper bound: 330.6217315
time: 8.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6215042
time: 8.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.66
Output dim: 9, lower bound: -330.6216385, upper bound: 330.6217315
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.66
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6215042

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -180.3420868, 143.4214325, -183.6371460, 146.0147095, -326.3567810, 327.0585327
1: -151.3821564, 127.4255600, -154.1186218, 129.7214966, -281.1036072, 281.5441589
2: -198.9765472, 128.9956970, -202.5823517, 131.3443756, -330.3209229, 331.5780029
3: -211.3465729, 111.4465866, -215.1947327, 113.4665604, -324.8131409, 326.6413269
4: -194.3313751, 148.9740448, -197.8540955, 151.6625977, -345.9939575, 346.8281250
5: -173.8146667, 135.1881104, -176.9648895, 137.6301270, -311.4447937, 312.1529846
6: -166.6844940, 159.9958649, -169.7169952, 162.8930969, -329.5775146, 329.7128296
7: -181.1339569, 151.9680023, -184.4264832, 154.7376404, -335.8715820, 336.3944702
8: -217.8480682, 148.5378723, -221.7955170, 151.2197571, -369.0678101, 370.3333740
9: -164.8039093, 162.5196381, -167.8070984, 165.4597626, -330.2636719, 330.3266602

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6215042
time: 6.39 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6215042
time: 7.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -186.7120209, 148.4876099, -183.1186218, 145.6068726, -332.3189087, 331.6062317
1: -156.7530975, 131.9496307, -153.6882477, 129.3605499, -286.1136169, 285.6378174
2: -206.0017548, 133.5013428, -202.0144958, 130.9743042, -336.9760742, 335.5158081
3: -218.9201202, 115.4685135, -214.5903778, 113.1497345, -332.0698242, 330.0588989
4: -201.1480560, 154.2209930, -197.2991943, 151.2396698, -352.3877258, 351.5200806
5: -179.9620514, 139.9226074, -176.4696655, 137.2460480, -317.2080688, 316.3922119
6: -172.5550537, 165.6704559, -169.2395935, 162.4378052, -334.9928589, 334.9100342
7: -187.4775543, 157.3350067, -183.9081268, 154.3020020, -341.7795410, 341.2431335
8: -225.5724030, 153.8252563, -221.1740875, 150.7976990, -376.3701172, 374.9993286
9: -170.5974579, 168.2449493, -167.3347626, 164.9983063, -335.5957031, 335.5796509

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6214980
time: 7.32 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6214980, upper bound: 330.6214980
time: 8.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.65
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6215042
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.65
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6215042
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.65
Output dim: 9, lower bound: -330.6215042, upper bound: 330.6214980
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.65
Output dim: 9, lower bound: -330.6214980, upper bound: 330.6214980

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -180.3420868, 143.4214325, -180.3420868, 143.4214325, -323.7634888, 323.7634888
1: -151.3821564, 127.4255600, -151.3821564, 127.4255600, -278.8076782, 278.8076782
2: -198.9765472, 128.9956970, -198.9765472, 128.9956970, -327.9721985, 327.9721985
3: -211.3465729, 111.4465866, -211.3465729, 111.4465866, -322.7931519, 322.7931519
4: -194.3313751, 148.9740448, -194.3313751, 148.9740448, -343.3054199, 343.3054199
5: -173.8146667, 135.1881104, -173.8146667, 135.1881104, -309.0027771, 309.0027771
6: -166.6844940, 159.9958649, -166.6844940, 159.9958649, -326.6802979, 326.6802979
7: -181.1339569, 151.9680023, -181.1339569, 151.9680023, -333.1019592, 333.1019592
8: -217.8480682, 148.5378723, -217.8480682, 148.5378723, -366.3858948, 366.3858948
9: -164.8039093, 162.5196381, -164.8039093, 162.5196381, -327.3235474, 327.3235474

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123602, upper bound: 330.6121899
time: 7.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6165424, upper bound: 330.6165769
time: 7.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -180.3420868, 143.4214325, -186.7120209, 148.4876099, -328.8297119, 330.1334534
1: -151.3821564, 127.4255600, -156.7530975, 131.9496307, -283.3316956, 284.1786499
2: -198.9765472, 128.9956970, -206.0017548, 133.5013428, -332.4778748, 334.9974365
3: -211.3465729, 111.4465866, -218.9201202, 115.4685135, -326.8150940, 330.3666382
4: -194.3313751, 148.9740448, -201.1480560, 154.2209930, -348.5522766, 350.1221008
5: -173.8146667, 135.1881104, -179.9620514, 139.9226074, -313.7372742, 315.1501465
6: -166.6844940, 159.9958649, -172.5550537, 165.6704559, -332.3549194, 332.5509033
7: -181.1339569, 151.9680023, -187.4775543, 157.3350067, -338.4689636, 339.4455566
8: -217.8480682, 148.5378723, -225.5724030, 153.8252563, -371.6733093, 374.1102600
9: -164.8039093, 162.5196381, -170.5974579, 168.2449493, -333.0488281, 333.1170654

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123602, upper bound: 330.6121899
time: 7.80 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6165424, upper bound: 330.6165769
time: 7.63 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -186.1888428, 148.0735931, -179.0932617, 142.4330597, -328.6218872, 327.1668091
1: -156.3160095, 131.5839844, -150.3538055, 126.5763321, -282.8923340, 281.9377747
2: -205.4256439, 133.1296234, -197.6128693, 128.1499176, -333.5755310, 330.7424927
3: -218.3047943, 115.1476364, -209.8979797, 110.7066879, -329.0114746, 325.0456238
4: -200.5850220, 153.7935028, -193.0086823, 147.9845886, -348.5696106, 346.8020935
5: -179.4581604, 139.5332336, -172.6100006, 134.2817230, -313.7398376, 312.1431885
6: -172.0746765, 165.2075958, -165.5950470, 158.9017487, -330.9764404, 330.8026428
7: -186.9512482, 156.8967438, -179.9202423, 150.9687500, -337.9199829, 336.8168945
8: -224.9434662, 153.4008331, -216.3416290, 147.5127716, -372.4561462, 369.7424622
9: -170.1219330, 167.7768860, -163.7141266, 161.4434814, -331.5654297, 331.4909973

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 8.02 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.20 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -186.7120209, 148.4876099, -182.1649780, 144.8529663, -331.5649719, 330.6525879
1: -156.7530975, 131.9496307, -152.8902130, 128.6935272, -285.4466248, 284.8398438
2: -206.0017548, 133.5013428, -200.9638519, 130.2978516, -336.2996216, 334.4652100
3: -218.9201202, 115.4685135, -213.4694519, 112.5672607, -331.4873657, 328.9379578
4: -201.1480560, 154.2209930, -196.2718964, 150.4602203, -351.6082764, 350.4927979
5: -179.9620514, 139.9226074, -175.5517120, 136.5379791, -316.5000305, 315.4743042
6: -172.5550537, 165.6704559, -168.3637695, 161.5925446, -334.1475830, 334.0342102
7: -187.4775543, 157.3350067, -182.9495087, 153.5032959, -340.9808350, 340.2845154
8: -225.5724030, 153.8252563, -220.0286560, 150.0230560, -375.5954590, 373.8538818
9: -170.5974579, 168.2449493, -166.4678040, 164.1441803, -334.7416382, 334.7127380

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 9.17 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6123602, upper bound: 330.6121899
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6165424, upper bound: 330.6165769
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6123602, upper bound: 330.6121899
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6165424, upper bound: 330.6165769
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -160.9380188, 127.9871902, -177.7893982, 141.3888397, -302.3268433, 305.7765198
1: -134.9591827, 113.6294250, -149.2173157, 125.6078186, -260.5670166, 262.8467407
2: -177.3978271, 115.1857910, -196.1348114, 127.1768036, -304.5746155, 311.3205261
3: -188.3767700, 99.4048767, -208.3223419, 109.8607559, -298.2375183, 307.7271729
4: -173.4401855, 132.9075317, -191.5837250, 146.8571320, -320.2972717, 324.4912720
5: -155.1359100, 120.5766220, -171.3547974, 133.2644196, -288.4003296, 291.9314270
6: -148.6571503, 142.7648010, -164.3094330, 157.7277527, -306.3848877, 307.0742188
7: -161.5084839, 135.5796204, -178.5491791, 149.8096771, -311.3181763, 314.1287842
8: -194.2768555, 132.5039368, -214.7450256, 146.4279480, -340.7047729, 347.2489014
9: -147.0541840, 145.0638123, -162.4677734, 160.2196350, -307.2738037, 307.5315552

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127595, upper bound: 330.6125432
time: 8.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
time: 8.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -180.1908722, 143.3010101, -311.8451538, 314.2220764
1: -141.3865662, 119.0408478, -151.2539520, 127.3179626, -268.7045288, 270.2947998
2: -185.8666382, 120.5881119, -198.8083954, 128.8878632, -314.7545166, 319.3965149
3: -197.4193420, 104.1297684, -211.1678772, 111.3527145, -308.7720642, 315.2976379
4: -181.6414948, 139.2112579, -194.1686859, 148.8488312, -330.4902954, 333.3799133
5: -162.4455872, 126.3240967, -173.6688080, 135.0744324, -297.5200195, 299.9929199
6: -155.7493744, 149.5174408, -166.5441589, 159.8614502, -315.6108398, 316.0615540
7: -169.2233276, 142.0249481, -180.9811249, 151.8404541, -321.0637512, 323.0060425
8: -203.5285492, 138.7832489, -217.6643982, 148.4127502, -351.9412537, 356.4476318
9: -154.0368805, 151.9107056, -164.6657715, 162.3835144, -316.4204102, 316.5764771

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170754
time: 7.14 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170754
time: 6.56 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -160.9380188, 127.9871902, -184.1520996, 146.4486389, -307.3865967, 312.1392822
1: -134.9591827, 113.6294250, -154.5810547, 130.1262512, -265.0854187, 268.2104797
2: -177.3978271, 115.1857910, -203.1506042, 131.6761475, -309.0739746, 318.3363953
3: -188.3767700, 99.4048767, -215.8871765, 113.8774719, -302.2542114, 315.2920532
4: -173.4401855, 132.9075317, -198.3916779, 152.0974426, -325.5375366, 331.2991943
5: -155.1359100, 120.5766220, -177.4944611, 137.9924927, -293.1283875, 298.0710754
6: -148.6571503, 142.7648010, -170.1728516, 163.3945465, -312.0516968, 312.9376526
7: -161.5084839, 135.5796204, -184.8847504, 155.1696930, -316.6781311, 320.4643555
8: -194.2768555, 132.5039368, -222.4584045, 151.7087555, -345.9855652, 354.9623108
9: -147.0541840, 145.0638123, -168.2537231, 165.9369812, -312.9911499, 313.3174744

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
time: 9.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
time: 7.78 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -186.5614471, 148.3678131, -316.9119568, 320.5926819
1: -141.3865662, 119.0408478, -156.6255646, 131.8425903, -273.2291260, 275.6663818
2: -185.8666382, 120.5881119, -205.8344116, 133.3941193, -319.2607117, 326.4225159
3: -197.4193420, 104.1297684, -218.7422333, 115.3751068, -312.7944336, 322.8720093
4: -181.6414948, 139.2112579, -200.9861603, 154.0963287, -335.7378235, 340.1974182
5: -162.4455872, 126.3240967, -179.8168793, 139.8094482, -302.2549744, 306.1409302
6: -155.7493744, 149.5174408, -172.4154205, 165.5367584, -321.2861328, 321.9328613
7: -169.2233276, 142.0249481, -187.3254242, 157.2081299, -326.4314270, 329.3503418
8: -203.5285492, 138.7832489, -225.3897095, 153.7007751, -357.2292786, 364.1729431
9: -154.0368805, 151.9107056, -170.4600067, 168.1094818, -322.1463623, 322.3706665

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122406, upper bound: 330.6124771
time: 6.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122406, upper bound: 330.6165769
time: 8.93 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -166.5543671, 132.4510040, -176.5265656, 140.3897552, -306.9440918, 308.9775696
1: -139.6953125, 117.6249847, -148.1768341, 124.7488022, -264.4441223, 265.8018188
2: -183.5892029, 119.1505051, -194.7554932, 126.3215027, -309.9107056, 313.9059448
3: -195.0648804, 102.9589462, -206.8575897, 109.1116638, -304.1765442, 309.8165283
4: -179.4498901, 137.5328979, -190.2460327, 145.8565216, -325.3063660, 327.7789307
5: -160.5511780, 124.7503891, -170.1373901, 132.3466797, -292.8978577, 294.8877869
6: -153.8300018, 147.7708130, -163.2077179, 156.6212616, -310.4512024, 310.9785156
7: -167.0906830, 140.3113556, -177.3223267, 148.7986603, -315.8892517, 317.6336670
8: -201.0901489, 137.1716156, -213.2215424, 145.3903961, -346.4805298, 350.3931274
9: -152.1632538, 150.1122131, -161.3649445, 159.1306458, -311.2938843, 311.4771729

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 7.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 8.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -174.3066101, 138.6127472, -178.9468842, 142.3165436, -316.6231689, 317.5596008
1: -146.2491455, 123.1382751, -150.2297821, 126.4722977, -272.7214355, 273.3680420
2: -192.2186890, 124.6620178, -197.4502106, 128.0456390, -320.2643433, 322.1122131
3: -204.2702332, 107.7747040, -209.7251587, 110.6159592, -314.8861694, 317.4998779
4: -187.8027039, 143.9607544, -192.8512421, 147.8634796, -335.6661072, 336.8119507
5: -168.0055542, 130.6056976, -172.4689331, 134.1717224, -302.1771851, 303.0746155
6: -161.0579834, 154.6533966, -165.4594421, 158.7717285, -319.8297119, 320.1127930
7: -174.9512177, 146.8813477, -179.7724152, 150.8453979, -325.7966003, 326.6537476
8: -210.5228729, 143.5730591, -216.1639862, 147.3917542, -357.9145813, 359.7370605
9: -159.2757111, 157.0884247, -163.5805206, 161.3118744, -320.5875549, 320.6689453

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 6.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 6.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -167.0620422, 132.8526154, -179.6320038, 142.8364868, -309.8984985, 312.4845886
1: -140.1192017, 117.9796524, -150.7423859, 126.8899765, -267.0091858, 268.7220459
2: -184.1480865, 119.5109940, -198.1441956, 128.4936523, -312.6417236, 317.6551819
3: -195.6618958, 103.2701416, -210.4687805, 110.9939117, -306.6557922, 313.7388611
4: -179.9962006, 137.9474030, -193.5455017, 148.3600159, -328.3561096, 331.4929199
5: -161.0397797, 125.1281433, -173.1116333, 134.6291046, -295.6688843, 298.2397766
6: -154.2957306, 148.2198486, -166.0079803, 159.3420563, -313.6377869, 314.2277832
7: -167.6010132, 140.7364502, -180.3855743, 151.3622589, -318.9631653, 321.1219788
8: -201.7003937, 137.5831451, -216.9492798, 147.9295807, -349.6299744, 354.5324097
9: -152.6244354, 150.5662994, -164.1500092, 161.8622742, -314.4866638, 314.7163086

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 7.80 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 7.50 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -174.8271179, 139.0247040, -182.0149841, 144.7335663, -319.5606689, 321.0396729
1: -146.6840363, 123.5020981, -152.7631378, 128.5868378, -275.2708740, 276.2652283
2: -192.7919464, 125.0317993, -200.7971344, 130.1909637, -322.9828491, 325.8289185
3: -204.8824310, 108.0939713, -213.2922974, 112.4742050, -317.3566284, 321.3862610
4: -188.3628540, 144.3860474, -196.1105347, 150.3360748, -338.6989136, 340.4965515
5: -168.5068359, 130.9931030, -175.4071503, 136.4252014, -304.9320374, 306.4002686
6: -161.5359650, 155.1138611, -168.2246857, 161.4592743, -322.9952087, 323.3385620
7: -175.4748840, 147.3174591, -182.7980194, 153.3768158, -328.8516541, 330.1154785
8: -211.1485291, 143.9952850, -219.8465118, 149.8990021, -361.0475464, 363.8417969
9: -159.7488251, 157.5541534, -166.3308258, 164.0092468, -323.7580261, 323.8849792

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.85 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6127595, upper bound: 330.6125432
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170754
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170754
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6122406, upper bound: 330.6124771
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6122406, upper bound: 330.6165769
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.85
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -160.4354095, 127.5897522, -173.7248230, 138.1838684, -298.6192017, 301.3145752
1: -134.5396576, 113.2785339, -145.8494720, 122.7952805, -257.3349304, 259.1279907
2: -176.8446198, 114.8290100, -191.6897583, 124.3241272, -301.1687622, 306.5187378
3: -187.7860718, 99.0969696, -203.5825653, 107.3925858, -295.1786499, 302.6794739
4: -172.8993683, 132.4973145, -187.2512970, 143.5694275, -316.4688110, 319.7485962
5: -154.6523285, 120.2027435, -167.4580841, 130.2697906, -284.9221191, 287.6607971
6: -148.1963196, 142.3203278, -160.6284790, 154.1569214, -302.3532410, 302.9487915
7: -161.0034790, 135.1590118, -174.5221863, 146.4431610, -307.4466553, 309.6812134
8: -193.6729736, 132.0966797, -209.8645325, 143.1097412, -336.7827148, 341.9611816
9: -146.5977631, 144.6144867, -158.8107758, 156.6284637, -303.2261658, 303.4252319

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
time: 7.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
time: 7.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -160.9380188, 127.9871902, -176.8511658, 140.6475830, -301.5855713, 304.8383484
1: -134.9591827, 113.6294250, -148.4328003, 124.9519730, -259.9111328, 262.0622253
2: -177.3978271, 115.1857910, -195.1016541, 126.5118408, -303.9096069, 310.2874451
3: -188.3767700, 99.4048767, -207.2198334, 109.2882309, -297.6649475, 306.6246643
4: -173.4401855, 132.9075317, -190.5731354, 146.0909119, -319.5310364, 323.4806519
5: -155.1359100, 120.5766220, -170.4525604, 132.5680389, -287.7039185, 291.0291748
6: -148.6571503, 142.7648010, -163.4487915, 156.8963928, -305.5535278, 306.2135620
7: -161.5084839, 135.5796204, -177.6070099, 149.0245972, -310.5330811, 313.1866455
8: -194.2768555, 132.5039368, -213.6180573, 145.6664276, -339.9432983, 346.1219177
9: -147.0541840, 145.0638123, -161.6152802, 159.3797760, -306.4339600, 306.6790771

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
time: 7.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
time: 4.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -168.0359344, 133.6294403, -176.1472168, 140.1122284, -308.1481628, 309.7766724
1: -140.9623718, 118.6860199, -147.9040070, 124.5201721, -265.4825439, 266.5900269
2: -185.3072205, 120.2274094, -194.3867493, 126.0497208, -311.3569031, 314.6141663
3: -196.8221893, 103.8183823, -206.4524689, 108.8980713, -305.7202454, 310.2708435
4: -181.0946960, 138.7964172, -189.8586273, 145.5779877, -326.6726685, 328.6550293
5: -161.9566650, 125.9460297, -169.7915192, 132.0963593, -294.0529785, 295.7375488
6: -155.2834320, 149.0679932, -162.8821564, 156.3091583, -311.5925598, 311.9501343
7: -168.7126312, 141.5995941, -176.9744720, 148.4916992, -317.2043152, 318.5740356
8: -202.9178925, 138.3714752, -212.8094177, 145.1126251, -348.0305176, 351.1807861
9: -153.5753174, 151.4564056, -161.0281525, 158.8114319, -312.3867493, 312.4845581

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
time: 7.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
time: 7.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -179.2355347, 142.5456238, -311.0897827, 313.2667847
1: -141.3865662, 119.0408478, -150.4546509, 126.6496735, -268.0362549, 269.4954834
2: -185.8666382, 120.5881119, -197.7559204, 128.2102051, -314.0768433, 318.3440247
3: -197.4193420, 104.1297684, -210.0446930, 110.7691956, -308.1885376, 314.1744690
4: -181.6414948, 139.2112579, -193.1395416, 148.0680084, -329.7095032, 332.3507690
5: -162.4455872, 126.3240967, -172.7493591, 134.3651123, -296.8106995, 299.0734558
6: -155.7493744, 149.5174408, -165.6667786, 159.0147095, -314.7640991, 315.1842041
7: -169.2233276, 142.0249481, -180.0208435, 151.0403442, -320.2635498, 322.0457458
8: -203.5285492, 138.7832489, -216.5168304, 147.6367798, -351.1653442, 355.3000183
9: -154.0368805, 151.9107056, -163.7972412, 161.5278473, -315.5647278, 315.7079468

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
time: 7.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170754
time: 7.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -160.4354095, 127.5897522, -179.8513794, 143.0554962, -303.4908447, 307.4411011
1: -134.5396576, 113.2785339, -151.0162811, 127.1471024, -261.6867676, 264.2947998
2: -176.8446198, 114.8290100, -198.4477386, 128.6610107, -305.5056152, 313.2767334
3: -187.7860718, 99.0969696, -210.8604126, 111.2612305, -299.0473022, 309.9573364
4: -172.8993683, 132.4973145, -193.8068848, 148.6119995, -321.5113525, 326.3041992
5: -154.6523285, 120.2027435, -173.3677979, 134.8270874, -289.4793701, 293.5705566
6: -148.1963196, 142.3203278, -166.2730255, 159.6150055, -307.8113098, 308.5933228
7: -161.0034790, 135.1590118, -180.6239166, 151.6080017, -312.6114502, 315.7829285
8: -193.6729736, 132.0966797, -217.3003235, 148.1896973, -341.8626709, 349.3970032
9: -146.5977631, 144.6144867, -164.3847656, 162.1357880, -308.7335205, 308.9992676

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
time: 8.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
time: 8.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -160.9380188, 127.9871902, -183.1782990, 145.6781311, -306.6161499, 311.1654968
1: -134.9591827, 113.6294250, -153.7655334, 129.4446106, -264.4038086, 267.3949585
2: -177.3978271, 115.1857910, -202.0773315, 130.9847870, -308.3826294, 317.2630920
3: -188.3767700, 99.4048767, -214.7415924, 113.2821732, -301.6589050, 314.1464539
4: -173.4401855, 132.9075317, -197.3424530, 151.3008881, -324.7410583, 330.2499695
5: -155.1359100, 120.5766220, -176.5564117, 137.2690735, -292.4049683, 297.1330261
6: -148.6571503, 142.7648010, -169.2774506, 162.5310822, -311.1882324, 312.0422363
7: -161.5084839, 135.5796204, -183.9050903, 154.3535004, -315.8619385, 319.4847107
8: -194.2768555, 132.5039368, -221.2883911, 150.9172516, -345.1940918, 353.7922668
9: -147.0541840, 145.0638123, -167.3678589, 165.0642242, -312.1184082, 312.4316711

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
time: 8.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
time: 8.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -167.0620422, 132.8526154, -301.3967590, 301.0932617
1: -141.3865662, 119.0408478, -140.1192017, 117.9796524, -259.3662109, 259.1600342
2: -185.8666382, 120.5881119, -184.1480865, 119.5109940, -305.3776245, 304.7361755
3: -197.4193420, 104.1297684, -195.6618958, 103.2701416, -300.6894836, 299.7916565
4: -181.6414948, 139.2112579, -179.9962006, 137.9474030, -319.5888977, 319.2074280
5: -162.4455872, 126.3240967, -161.0397797, 125.1281433, -287.5737000, 287.3638611
6: -155.7493744, 149.5174408, -154.2957306, 148.2198486, -303.9692383, 303.8131409
7: -169.2233276, 142.0249481, -167.6010132, 140.7364502, -309.9597473, 309.6258240
8: -203.5285492, 138.7832489, -201.7003937, 137.5831451, -341.1116333, 340.4836426
9: -154.0368805, 151.9107056, -152.6244354, 150.5662994, -304.6031799, 304.5350952

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6124735
time: 8.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6124735
time: 8.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -174.8271179, 139.0247040, -307.5688477, 308.8583374
1: -141.3865662, 119.0408478, -146.6840363, 123.5020981, -264.8886719, 265.7248840
2: -185.8666382, 120.5881119, -192.7919464, 125.0317993, -310.8984375, 313.3799744
3: -197.4193420, 104.1297684, -204.8824310, 108.0939713, -305.5133057, 309.0122070
4: -181.6414948, 139.2112579, -188.3628540, 144.3860474, -326.0275269, 327.5740662
5: -162.4455872, 126.3240967, -168.5068359, 130.9931030, -293.4386902, 294.8309326
6: -155.7493744, 149.5174408, -161.5359650, 155.1138611, -310.8632202, 311.0533752
7: -169.2233276, 142.0249481, -175.4748840, 147.3174591, -316.5407715, 317.4997864
8: -203.5285492, 138.7832489, -211.1485291, 143.9952850, -347.5238342, 349.9317322
9: -154.0368805, 151.9107056, -159.7488251, 157.5541534, -311.5910339, 311.6594849

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6165640
time: 7.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6165640
time: 7.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -166.5543671, 132.4510040, -173.7248230, 138.1838684, -304.7381897, 306.1758423
1: -139.6953125, 117.6249847, -145.8494720, 122.7952805, -262.4906006, 263.4744263
2: -183.5892029, 119.1505051, -191.6897583, 124.3241272, -307.9133301, 310.8402100
3: -195.0648804, 102.9589462, -203.5825653, 107.3925858, -302.4574585, 306.5414734
4: -179.4498901, 137.5328979, -187.2512970, 143.5694275, -323.0192871, 324.7841797
5: -160.5511780, 124.7503891, -167.4580841, 130.2697906, -290.8209839, 292.2084351
6: -153.8300018, 147.7708130, -160.6284790, 154.1569214, -307.9869080, 308.3992920
7: -167.0906830, 140.3113556, -174.5221863, 146.4431610, -313.5338440, 314.8335571
8: -201.0901489, 137.1716156, -209.8645325, 143.1097412, -344.1998901, 347.0361023
9: -152.1632538, 150.1122131, -158.8107758, 156.6284637, -308.7916260, 308.9229736

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 7.47 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 7.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -166.5543671, 132.4510040, -179.7749023, 142.9926453, -309.5469666, 312.2258606
1: -139.6953125, 117.6249847, -150.9497528, 127.0915375, -266.7868652, 268.5747375
2: -183.5892029, 119.1505051, -198.3615723, 128.6044006, -312.1935730, 317.5120544
3: -195.0648804, 102.9589462, -210.7677917, 111.2122879, -306.2771606, 313.7267151
4: -179.4498901, 137.5328979, -193.7238617, 148.5467224, -327.9965820, 331.2566833
5: -160.5511780, 124.7503891, -173.2907104, 134.7696228, -295.3207397, 298.0411072
6: -153.8300018, 147.7708130, -166.1981049, 159.5462036, -313.3762207, 313.9689026
7: -167.0906830, 140.3113556, -180.5430145, 151.5411835, -318.6318665, 320.8543701
8: -201.0901489, 137.1716156, -217.2082367, 148.1251526, -349.2153015, 354.3798218
9: -152.1632538, 150.1122131, -164.3132782, 162.0650940, -314.2283020, 314.4254761

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 8.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 8.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -174.3066101, 138.6127472, -176.1472168, 140.1122284, -314.4188232, 314.7599487
1: -146.2491455, 123.1382751, -147.9040070, 124.5201721, -270.7693176, 271.0422974
2: -192.2186890, 124.6620178, -194.3867493, 126.0497208, -318.2683716, 319.0487671
3: -204.2702332, 107.7747040, -206.4524689, 108.8980713, -313.1682739, 314.2271729
4: -187.8027039, 143.9607544, -189.8586273, 145.5779877, -333.3806763, 333.8193970
5: -168.0055542, 130.6056976, -169.7915192, 132.0963593, -300.1018372, 300.3972168
6: -161.0579834, 154.6533966, -162.8821564, 156.3091583, -317.3671265, 317.5355530
7: -174.9512177, 146.8813477, -176.9744720, 148.4916992, -323.4428711, 323.8558350
8: -210.5228729, 143.5730591, -212.8094177, 145.1126251, -355.6354980, 356.3824768
9: -159.2757111, 157.0884247, -161.0281525, 158.8114319, -318.0871277, 318.1165771

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -174.3066101, 138.6127472, -182.2041626, 144.9265594, -319.2331543, 320.8168945
1: -146.2491455, 123.1382751, -153.0106812, 128.8218384, -275.0709839, 276.1489563
2: -192.2186890, 124.6620178, -201.0664368, 130.3347931, -322.5534668, 325.7284546
3: -204.2702332, 107.7747040, -213.6451874, 112.7221069, -316.9923096, 321.4198914
4: -187.8027039, 143.9607544, -196.3394012, 150.5613708, -338.3640747, 340.3001404
5: -168.0055542, 130.6056976, -175.6303558, 136.6014557, -304.6069641, 306.2360535
6: -161.0579834, 154.6533966, -168.4581299, 161.7053986, -322.7633667, 323.1115112
7: -174.9512177, 146.8813477, -183.0024872, 153.5955048, -328.5466919, 329.8838196
8: -210.5228729, 143.5730591, -220.1617737, 150.1340027, -360.6568604, 363.7348328
9: -159.2757111, 157.0884247, -166.5370178, 164.2545471, -323.5302429, 323.6254272

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 6.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -167.0620422, 132.8526154, -176.8511658, 140.6475830, -307.7095947, 309.7037659
1: -140.1192017, 117.9796524, -148.4328003, 124.9519730, -265.0711670, 266.4124451
2: -184.1480865, 119.5109940, -195.1016541, 126.5118408, -310.6598816, 314.6126404
3: -195.6618958, 103.2701416, -207.2198334, 109.2882309, -304.9500732, 310.4899292
4: -179.9962006, 137.9474030, -190.5731354, 146.0909119, -326.0870972, 328.5205383
5: -161.0397797, 125.1281433, -170.4525604, 132.5680389, -293.6077881, 295.5806885
6: -154.2957306, 148.2198486, -163.4487915, 156.8963928, -311.1921387, 311.6686401
7: -167.6010132, 140.7364502, -177.6070099, 149.0245972, -316.6255493, 318.3434448
8: -201.7003937, 137.5831451, -213.6180573, 145.6664276, -347.3668213, 351.2011414
9: -152.6244354, 150.5662994, -161.6152802, 159.3797760, -312.0042114, 312.1815796

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 7.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 7.20 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -167.0620422, 132.8526154, -183.0750275, 145.5932922, -312.6553345, 315.9275818
1: -140.1192017, 117.9796524, -153.6757507, 129.3695679, -269.4887695, 271.6553955
2: -184.1480865, 119.5109940, -201.9608765, 130.9083405, -315.0563660, 321.4718628
3: -195.6618958, 103.2701416, -214.6165924, 113.2160797, -308.8779907, 317.8866882
4: -179.9962006, 137.9474030, -197.2303467, 151.2127533, -331.2089233, 335.1777344
5: -161.0397797, 125.1281433, -176.4524384, 137.1914673, -298.2312622, 301.5805359
6: -154.2957306, 148.2198486, -169.1763916, 162.4382172, -316.7338867, 317.3962402
7: -167.6010132, 140.7364502, -183.7958374, 154.2632294, -321.8641663, 324.5322266
8: -201.7003937, 137.5831451, -221.1639709, 150.8301697, -352.5305786, 358.7470398
9: -152.6244354, 150.5662994, -167.2713623, 164.9687195, -317.5931091, 317.8376465

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 8.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
time: 8.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -174.8271179, 139.0247040, -179.2355347, 142.5456238, -317.3727417, 318.2602234
1: -146.6840363, 123.5020981, -150.4546509, 126.6496735, -273.3337097, 273.9567261
2: -192.7919464, 125.0317993, -197.7559204, 128.2102051, -321.0021362, 322.7877197
3: -204.8824310, 108.0939713, -210.0446930, 110.7691956, -315.6516113, 318.1386719
4: -188.3628540, 144.3860474, -193.1395416, 148.0680084, -336.4308472, 337.5255737
5: -168.5068359, 130.9931030, -172.7493591, 134.3651123, -302.8719482, 303.7424622
6: -161.5359650, 155.1138611, -165.6667786, 159.0147095, -320.5506592, 320.7806396
7: -175.4748840, 147.3174591, -180.0208435, 151.0403442, -326.5151367, 327.3383179
8: -211.1485291, 143.9952850, -216.5168304, 147.6367798, -358.7853088, 360.5121155
9: -159.7488251, 157.5541534, -163.7972412, 161.5278473, -321.2766113, 321.3513794

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 10.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -174.8271179, 139.0247040, -185.4646301, 147.4962616, -322.3233643, 324.4892883
1: -146.6840363, 123.5020981, -155.7030945, 131.0715332, -277.7555542, 279.2051697
2: -192.7919464, 125.0317993, -204.6223907, 132.6117096, -325.4036255, 329.6541748
3: -204.8824310, 108.0939713, -217.4477234, 114.7010880, -319.5835266, 325.5416870
4: -188.3628540, 144.3860474, -199.8034058, 153.1947479, -341.5575256, 344.1894531
5: -168.5068359, 130.9931030, -178.7550049, 138.9935913, -307.5003967, 309.7481079
6: -161.5359650, 155.1138611, -171.3996735, 164.5626221, -326.0985718, 326.5135193
7: -175.4748840, 147.3174591, -186.2156067, 156.2844086, -331.7592773, 333.5330811
8: -211.1485291, 143.9952850, -224.0715179, 152.8054962, -363.9540405, 368.0668030
9: -159.7488251, 157.5541534, -169.4591675, 167.1230164, -326.8718262, 327.0133057

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
time: 7.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.18 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6127572, upper bound: 330.6125432
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170754
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6124735
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6124735
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6165640
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6165640
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.18
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -156.8526611, 124.7644653, -173.7248230, 138.1838684, -295.0364990, 298.4892883
1: -131.5747223, 110.8021164, -145.8494720, 122.7952805, -254.3699799, 256.6515503
2: -172.9314728, 112.3184738, -191.6897583, 124.3241272, -297.2556152, 304.0081787
3: -183.6140442, 96.9212494, -203.5825653, 107.3925858, -291.0066223, 300.5038147
4: -169.0901489, 129.6018372, -187.2512970, 143.5694275, -312.6595764, 316.8531189
5: -151.2193298, 117.5645828, -167.4580841, 130.2697906, -281.4891357, 285.0226746
6: -144.9535065, 139.1769104, -160.6284790, 154.1569214, -299.1104126, 299.8053894
7: -157.4647827, 132.1934662, -174.5221863, 146.4431610, -303.9079590, 306.7156372
8: -189.3725433, 129.1676636, -209.8645325, 143.1097412, -332.4822998, 339.0321350
9: -143.3786774, 141.4527283, -158.8107758, 156.6284637, -300.0070801, 300.2634888

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6103653
time: 7.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6125432
time: 7.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -160.0070801, 127.2517471, -173.7248230, 138.1838684, -298.1908875, 300.9765625
1: -134.1806183, 112.9787064, -145.8494720, 122.7952805, -256.9758911, 258.8281860
2: -176.3727112, 114.5259705, -191.6897583, 124.3241272, -300.6968384, 306.2156982
3: -187.2828979, 98.8370590, -203.5825653, 107.3925858, -294.6754761, 302.4195557
4: -172.4374695, 132.1474609, -187.2512970, 143.5694275, -316.0068970, 319.3987427
5: -154.2406616, 119.8856354, -167.4580841, 130.2697906, -284.5104370, 287.3436890
6: -147.8032684, 141.9399109, -160.6284790, 154.1569214, -301.9602051, 302.5683899
7: -160.5737305, 134.8005829, -174.5221863, 146.4431610, -307.0169067, 309.3227539
8: -193.1587830, 131.7485504, -209.8645325, 143.1097412, -336.2685242, 341.6130676
9: -146.2083740, 144.2302856, -158.8107758, 156.6284637, -302.8368530, 303.0410461

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6103653
time: 7.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6125432
time: 7.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -156.8526611, 124.7644653, -176.8511658, 140.6475830, -297.5002136, 301.6156311
1: -131.5747223, 110.8021164, -148.4328003, 124.9519730, -256.5266724, 259.2349243
2: -172.9314728, 112.3184738, -195.1016541, 126.5118408, -299.4432678, 307.4201050
3: -183.6140442, 96.9212494, -207.2198334, 109.2882309, -292.9021912, 304.1410828
4: -169.0901489, 129.6018372, -190.5731354, 146.0909119, -315.1810608, 320.1749268
5: -151.2193298, 117.5645828, -170.4525604, 132.5680389, -283.7873535, 288.0171509
6: -144.9535065, 139.1769104, -163.4487915, 156.8963928, -301.8499146, 302.6257019
7: -157.4647827, 132.1934662, -177.6070099, 149.0245972, -306.4893799, 309.8004761
8: -189.3725433, 129.1676636, -213.6180573, 145.6664276, -335.0389709, 342.7856445
9: -143.3786774, 141.4527283, -161.6152802, 159.3797760, -302.7584534, 303.0679932

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6103431
time: 7.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6125432
time: 8.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -160.0070801, 127.2517471, -176.8511658, 140.6475830, -300.6546021, 304.1029053
1: -134.1806183, 112.9787064, -148.4328003, 124.9519730, -259.1325378, 261.4114990
2: -176.3727112, 114.5259705, -195.1016541, 126.5118408, -302.8844910, 309.6276245
3: -187.2828979, 98.8370590, -207.2198334, 109.2882309, -296.5710754, 306.0568542
4: -172.4374695, 132.1474609, -190.5731354, 146.0909119, -318.5283508, 322.7205811
5: -154.2406616, 119.8856354, -170.4525604, 132.5680389, -286.8087158, 290.3381958
6: -147.8032684, 141.9399109, -163.4487915, 156.8963928, -304.6996460, 305.3887024
7: -160.5737305, 134.8005829, -177.6070099, 149.0245972, -309.5983276, 312.4075928
8: -193.1587830, 131.7485504, -213.6180573, 145.6664276, -338.8251953, 345.3666077
9: -146.2083740, 144.2302856, -161.6152802, 159.3797760, -305.5881348, 305.8455811

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6103431
time: 8.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6125432
time: 10.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -164.8865662, 131.1536713, -176.1472168, 140.1122284, -304.9987793, 307.3009033
1: -138.3654785, 116.5219421, -147.9040070, 124.5201721, -262.8856506, 264.4259644
2: -181.8766327, 118.0279694, -194.3867493, 126.0497208, -307.9263611, 312.4147339
3: -193.1633453, 101.9182510, -206.4524689, 108.8980713, -302.0614014, 308.3707275
4: -177.7477722, 136.2639771, -189.8586273, 145.5779877, -323.3257446, 326.1225891
5: -158.9423065, 123.6379471, -169.7915192, 132.0963593, -291.0386353, 293.4294739
6: -152.4489899, 146.3106689, -162.8821564, 156.3091583, -308.7580566, 309.1928101
7: -165.6085205, 139.0061340, -176.9744720, 148.4916992, -314.1002197, 315.9805908
8: -199.1442871, 135.8033752, -212.8094177, 145.1126251, -344.2568970, 348.6127319
9: -150.7532806, 148.6904144, -161.0281525, 158.8114319, -309.5646973, 309.7185669

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125701, upper bound: 330.6127981
time: 7.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6125701, upper bound: 330.6170701
time: 8.43 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 17.35 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6103653
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6125432
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6103653
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103444, upper bound: 330.6125432
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6103431
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6125432
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6103431
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6103431, upper bound: 330.6125432
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6125701, upper bound: 330.6127981
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.35
Output dim: 9, lower bound: -330.6125701, upper bound: 330.6170701
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170753
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6170754, upper bound: 330.6170754
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6123528, upper bound: 330.6121719
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6124735
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6124735
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6165640
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122317, upper bound: 330.6165640
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6122671, upper bound: 330.6120778
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.35
Output dim: 9, lower bound: -330.6163730, upper bound: 330.6163730
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=333.2668151855469
rel_dist={9: [-330.6221776084849, 330.6221776084849]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6214224, upper bound: 330.6214785
time: 7.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213501, upper bound: 330.6213501
time: 8.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.73
Output dim: 9, lower bound: -330.6214224, upper bound: 330.6214785
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.73
Output dim: 9, lower bound: -330.6213501, upper bound: 330.6213501

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -180.3420868, 143.4214325, -182.4643097, 145.0915680, -325.4336548, 325.8856812
1: -151.3821564, 127.4255600, -153.1445312, 128.9042664, -280.2863159, 280.5700989
2: -198.9765472, 128.9956970, -201.2988892, 130.5082245, -329.4847717, 330.2945862
3: -211.3465729, 111.4465866, -213.8249817, 112.7474976, -324.0940552, 325.2715759
4: -194.3313751, 148.9740448, -196.6001892, 150.7056122, -345.0369568, 345.5742188
5: -173.8146667, 135.1881104, -175.8435364, 136.7607117, -310.5753784, 311.0316162
6: -166.6844940, 159.9958649, -168.6374969, 161.8618469, -328.5463257, 328.6333313
7: -181.1339569, 151.9680023, -183.2543945, 153.7516327, -334.8855896, 335.2224121
8: -217.8480682, 148.5378723, -220.3903809, 150.2651825, -368.1132202, 368.9282227
9: -164.8039093, 162.5196381, -166.7379150, 164.4131317, -329.2170410, 329.2574768

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6214216, upper bound: 330.6214699
time: 7.99 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6214200, upper bound: 330.6214699
time: 9.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -186.7120209, 148.4876099, -182.0151215, 144.7391205, -331.4511414, 330.5027466
1: -156.7530975, 131.9496307, -152.7724304, 128.5924225, -285.3455200, 284.7220154
2: -206.0017548, 133.5013428, -200.8060455, 130.1868896, -336.1886597, 334.3073730
3: -218.9201202, 115.4685135, -213.3044281, 112.4756393, -331.3957520, 328.7729492
4: -201.1480560, 154.2209930, -196.1183319, 150.3396759, -351.4877319, 350.3392334
5: -179.9620514, 139.9226074, -175.4157715, 136.4289246, -316.3909912, 315.3383789
6: -172.5550537, 165.6704559, -168.2237396, 161.4689789, -334.0240479, 333.8941956
7: -187.4775543, 157.3350067, -182.8049927, 153.3751373, -340.8526917, 340.1400146
8: -225.5724030, 153.8252563, -219.8517456, 149.8996735, -375.4720764, 373.6770020
9: -170.5974579, 168.2449493, -166.3298798, 164.0164948, -334.6139526, 334.5747681

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213501, upper bound: 330.6213474
time: 8.55 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6213474, upper bound: 330.6213474
time: 8.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.57
Output dim: 9, lower bound: -330.6214216, upper bound: 330.6214699
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.57
Output dim: 9, lower bound: -330.6214200, upper bound: 330.6214699
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.57
Output dim: 9, lower bound: -330.6213501, upper bound: 330.6213474
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.57
Output dim: 9, lower bound: -330.6213474, upper bound: 330.6213474

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -178.9425659, 142.3142242, -178.4209747, 141.9033966, -320.8459167, 320.7351685
1: -150.2134857, 126.4480057, -149.7950897, 126.1073914, -276.3208618, 276.2431030
2: -197.4356842, 128.0017700, -196.8775787, 127.6707077, -325.1063843, 324.8793030
3: -209.7015076, 110.5886688, -209.1114044, 110.2934494, -319.9949646, 319.7000122
4: -192.8254547, 147.8309479, -192.2904205, 147.4356079, -340.2610474, 340.1213684
5: -172.4672241, 134.1470642, -171.9666595, 133.7829285, -306.2500916, 306.1137085
6: -165.4002075, 158.7579498, -164.9762421, 158.3100891, -323.7102356, 323.7341919
7: -179.7267303, 150.7961121, -179.2483215, 150.4032440, -330.1299744, 330.0443115
8: -216.1663971, 147.4030457, -215.5362244, 146.9655151, -363.1318054, 362.9392700
9: -163.5324860, 161.2682190, -163.1007233, 160.8420410, -324.3745117, 324.3689270

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114230, upper bound: 330.6111109
time: 8.88 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163096, upper bound: 330.6163100
time: 7.23 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -180.1746368, 143.2889557, -181.5094604, 144.3366547, -324.5112915, 324.7984009
1: -151.2419739, 127.3083649, -152.3455505, 128.2364044, -279.4783936, 279.6539307
2: -198.7920074, 128.8768616, -200.2469330, 129.8309174, -328.6229248, 329.1237793
3: -211.1496429, 111.3442307, -212.7025604, 112.1643143, -323.3139648, 324.0467529
4: -194.1509552, 148.8371124, -195.5715485, 149.9252167, -344.0761414, 344.4086304
5: -173.6533813, 135.0637512, -174.9244995, 136.0518036, -309.7051392, 309.9882202
6: -166.5305481, 159.8474274, -167.7605438, 161.0155487, -327.5460510, 327.6079407
7: -180.9654999, 151.8276825, -182.2945862, 152.9519196, -333.9174194, 334.1222229
8: -217.6468964, 148.4018097, -219.2434540, 149.4895477, -367.1364441, 367.6452637
9: -164.6516113, 162.3695984, -165.8698730, 163.5579376, -328.2095337, 328.2394714

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114230, upper bound: 330.6111109
time: 7.91 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6163095, upper bound: 330.6163100
time: 8.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -185.2956543, 147.3666534, -178.0144501, 141.5845184, -326.8801880, 325.3811035
1: -155.5698090, 130.9597626, -149.4585266, 125.8249664, -281.3947754, 280.4182434
2: -204.4420166, 132.4949646, -196.4316559, 127.3801804, -331.8221741, 328.9266052
3: -217.2542114, 114.5997772, -208.6398163, 110.0474625, -327.3016663, 323.2395630
4: -199.6238098, 153.0636444, -191.8541107, 147.1046906, -346.7285156, 344.9176941
5: -178.5978851, 138.8685913, -171.5797729, 133.4830627, -312.0809021, 310.4483643
6: -171.2544861, 164.4174347, -164.6017609, 157.9544525, -329.2089233, 329.0191956
7: -186.0527649, 156.1484833, -178.8417358, 150.0625763, -336.1153259, 334.9901123
8: -223.8696747, 152.6763153, -215.0483704, 146.6345978, -370.5041809, 367.7246704
9: -169.3102264, 166.9777985, -162.7315369, 160.4833374, -329.7934875, 329.7093506

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6113733, upper bound: 330.6110590
time: 8.48 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6162009, upper bound: 330.6162009
time: 7.72 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -186.5436096, 148.3543549, -181.0643921, 143.9874725, -330.5310669, 329.4187622
1: -156.6120605, 131.8317261, -151.9768677, 127.9274216, -284.5394897, 283.8085632
2: -205.8161011, 133.3817902, -199.7586212, 129.5124817, -335.3285522, 333.1404114
3: -218.7219849, 115.3655472, -212.1868591, 111.8949661, -330.6169128, 327.5523987
4: -200.9665985, 154.0832062, -195.0942078, 149.5626373, -350.5292358, 349.1773682
5: -179.7998199, 139.7974396, -174.5007019, 135.7230072, -315.5228271, 314.2981567
6: -172.4002228, 165.5211334, -167.3506012, 160.6262970, -333.0265198, 332.8717346
7: -187.3080750, 157.1938477, -181.8493195, 152.5788422, -339.8869019, 339.0431519
8: -225.3700562, 153.6883545, -218.7098236, 149.1273956, -374.4974365, 372.3981934
9: -170.4442291, 168.0939941, -165.4655457, 163.1650085, -333.6091919, 333.5595398

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6113733, upper bound: 330.6110590
time: 8.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6162009, upper bound: 330.6162009
time: 7.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6114230, upper bound: 330.6111109
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6163096, upper bound: 330.6163100
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6114230, upper bound: 330.6111109
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6163095, upper bound: 330.6163100
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6113733, upper bound: 330.6110590
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6162009, upper bound: 330.6162009
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6113733, upper bound: 330.6110590
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.37
Output dim: 9, lower bound: -330.6162009, upper bound: 330.6162009

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -159.5770721, 126.9109879, -170.0234680, 135.2200012, -294.7970581, 296.9344482
1: -133.8231812, 112.6792679, -142.6755066, 120.1294022, -253.9525757, 255.3547668
2: -175.8998413, 114.2197266, -187.5315552, 121.6897736, -297.5895386, 301.7512817
3: -186.7772217, 98.5710449, -199.1648865, 105.0760880, -291.8533020, 297.7359009
4: -171.9758148, 131.7967987, -183.2512970, 140.4749298, -312.4507446, 315.0480957
5: -153.8264008, 119.5642624, -163.8795013, 127.4535217, -281.2798767, 283.4437561
6: -147.4093170, 141.5612946, -157.1655884, 150.8503571, -298.2596436, 298.7268677
7: -160.1409454, 134.4406433, -170.7498627, 143.3050079, -303.4459534, 305.1904907
8: -192.6416321, 131.4011841, -205.3268127, 140.0243988, -332.6660156, 336.7279968
9: -145.8183289, 143.8471069, -155.4153900, 153.2780914, -299.0964355, 299.2624512

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6057750, upper bound: 330.6055138
time: 7.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6054248, upper bound: 330.6052625
time: 8.80 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -167.1676788, 132.9429016, -173.9838257, 138.3730621, -305.5407410, 306.9267273
1: -140.2376862, 118.0797958, -146.0362854, 122.9560242, -263.1936340, 264.1160278
2: -184.3514862, 119.6112061, -191.9479980, 124.5092468, -308.8607178, 311.5591431
3: -195.8020172, 103.2863617, -203.8738708, 107.5427551, -303.3446350, 307.1602173
4: -180.1605988, 138.0876770, -187.5195465, 143.7641602, -323.9247437, 325.6072388
5: -161.1212616, 125.3000870, -167.6909943, 130.4486847, -291.5699463, 292.9910889
6: -154.4873657, 148.3001862, -160.8650360, 154.3697510, -308.8571167, 309.1651917
7: -167.8401184, 140.8729248, -174.7691040, 146.6647491, -314.5048218, 315.6419678
8: -201.8746338, 137.6679535, -210.1518402, 143.2972717, -345.1719055, 347.8197937
9: -152.7868958, 150.6802673, -159.0520630, 156.8527832, -309.6396484, 309.7323303

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
time: 7.56 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -160.7768402, 127.8598404, -173.1671753, 137.6980743, -298.4749146, 301.0269165
1: -134.8243713, 113.5167389, -145.2741394, 122.2975616, -257.1219482, 258.7908936
2: -177.2203369, 115.0715408, -190.9630127, 123.8901596, -301.1105042, 306.0344543
3: -188.1873169, 99.3065262, -202.8223267, 106.9848480, -295.1721802, 302.1288452
4: -173.2665253, 132.7759094, -186.5915222, 143.0115967, -316.2781067, 319.3674316
5: -154.9808807, 120.4569473, -166.8905792, 129.7671814, -284.7480469, 287.3475342
6: -148.5092926, 142.6219482, -160.0040283, 153.6049957, -302.1142883, 302.6259766
7: -161.3466034, 135.4447327, -173.8511200, 145.9024963, -307.2490845, 309.2958374
8: -194.0832367, 132.3731689, -209.1033325, 142.5950928, -336.6783447, 341.4765015
9: -146.9077301, 144.9194794, -158.2370300, 156.0463409, -302.9540710, 303.1564941

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6057750, upper bound: 330.6055138
time: 8.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6054248, upper bound: 330.6052625
time: 9.07 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.3814392, 133.9026794, -176.9399414, 140.6990662, -309.0805054, 310.8426208
1: -141.2505035, 118.9271164, -148.4732513, 124.9871674, -266.2376709, 267.4003296
2: -185.6874695, 120.4728012, -195.1676025, 126.5748291, -312.2622681, 315.6403503
3: -197.2281494, 104.0304718, -207.3049469, 109.3292389, -306.5573730, 311.3354187
4: -181.4662628, 139.0783997, -190.6564484, 146.1423035, -327.6085815, 329.7347412
5: -162.2891235, 126.2032852, -170.5200348, 132.6168365, -294.9059448, 296.7233276
6: -155.6001587, 149.3732758, -163.5229797, 156.9561920, -312.5563354, 312.8962402
7: -169.0599213, 141.8887939, -177.6795502, 149.0992432, -318.1591797, 319.5682678
8: -203.3331299, 138.6512146, -213.6951904, 145.7108917, -349.0440063, 352.3462830
9: -153.8890381, 151.7650604, -161.6981659, 159.4469452, -313.3359985, 313.4631958

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
time: 7.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
time: 6.83 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -165.6880798, 131.7657318, -169.6147461, 134.8994904, -300.5875854, 301.3804932
1: -138.9720154, 117.0198975, -142.3371735, 119.8454819, -258.8174744, 259.3570557
2: -182.6355591, 118.5354385, -187.0831757, 121.3977509, -304.0332642, 305.6186218
3: -194.0462341, 102.4279709, -198.6907043, 104.8288193, -298.8750610, 301.1186523
4: -178.5177612, 136.8256378, -182.8125763, 140.1422272, -318.6599731, 319.6382141
5: -159.7173920, 124.1058273, -163.4906311, 127.1520996, -286.8695068, 287.5964661
6: -153.0352936, 147.0047302, -156.7889709, 150.4928436, -303.5281067, 303.7937012
7: -166.2199554, 139.5861359, -170.3410339, 142.9625702, -309.1825256, 309.9271851
8: -200.0487976, 136.4695892, -204.8362885, 139.6917114, -339.7405090, 341.3058777
9: -151.3764038, 149.3375092, -155.0443115, 152.9175720, -304.2939148, 304.3818359

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6056999, upper bound: 330.6054159
time: 9.01 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6053305, upper bound: 330.6051665
time: 7.38 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -173.4177399, 137.9092407, -173.5765991, 138.0535889, -311.4713135, 311.4858398
1: -145.5064392, 122.5169373, -145.6991119, 122.6731186, -268.1795654, 268.2160645
2: -191.2397156, 124.0304794, -191.5013580, 124.2181931, -315.4578552, 315.5318298
3: -203.2246094, 107.2294693, -203.4014740, 107.2963409, -310.5209045, 310.6309509
4: -186.8460541, 143.2343750, -187.0824738, 143.4327393, -330.2788086, 330.3168335
5: -167.1494141, 129.9441376, -167.3034210, 130.1483154, -297.2977295, 297.2475586
6: -160.2416840, 153.8669434, -160.4899597, 154.0134888, -314.2551880, 314.3569031
7: -174.0569000, 146.1366425, -174.3618011, 146.3234406, -320.3803101, 320.4983826
8: -209.4542542, 142.8521118, -209.6632233, 142.9658203, -352.4200745, 352.5153198
9: -158.4678192, 156.2931366, -158.6822205, 156.4935303, -314.9613647, 314.9753418

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
time: 8.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571
time: 12.24 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -166.8987427, 132.7235107, -172.7201996, 137.3473663, -304.2460938, 305.4437256
1: -139.9825745, 117.8654633, -144.9038544, 121.9872284, -261.9697876, 262.7693176
2: -183.9682159, 119.3952103, -190.4723663, 123.5704651, -307.5386353, 309.8675537
3: -195.4699402, 103.1704636, -202.3043060, 106.7143707, -302.1843262, 305.4747620
4: -179.8202820, 137.8140106, -186.1119080, 142.6474609, -322.4677124, 323.9259033
5: -160.8826599, 125.0068512, -166.4650116, 129.4369659, -290.3196411, 291.4718628
6: -154.1458435, 148.0751343, -159.5921631, 153.2140350, -307.3598633, 307.6672974
7: -167.4369659, 140.5997009, -173.4038544, 145.5278320, -312.9647827, 314.0035400
8: -201.5042572, 137.4505768, -208.5672455, 142.2312164, -343.7354431, 346.0177917
9: -152.4760132, 150.4200439, -157.8309174, 155.6516724, -308.1276855, 308.2509766

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6056998, upper bound: 330.6054159
time: 8.79 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6053305, upper bound: 330.6051665
time: 9.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -174.6586761, 138.8913879, -176.4936066, 140.3489380, -315.0075989, 315.3850098
1: -146.5429688, 123.3842010, -148.1035309, 124.6772995, -271.2202759, 271.4877319
2: -192.6062469, 124.9122391, -194.6778870, 126.2554321, -318.8616333, 319.5900879
3: -204.6843109, 107.9909973, -206.7878571, 109.0591278, -313.7434387, 314.7788696
4: -188.1813965, 144.2482758, -190.1777496, 145.7787323, -333.9601440, 334.4260254
5: -168.3445587, 130.8679657, -170.0950317, 132.2871094, -300.6316528, 300.9630127
6: -161.3811035, 154.9645233, -163.1118469, 156.5658569, -317.9468994, 318.0763550
7: -175.3054047, 147.1762695, -177.2330322, 148.7250671, -324.0304565, 324.4092712
8: -210.9461823, 143.8583984, -213.1600037, 145.3477173, -356.2938843, 357.0184021
9: -159.5955658, 157.4031982, -161.2926941, 159.0528717, -318.6484375, 318.6958923

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
time: 8.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571
time: 6.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.78 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6057750, upper bound: 330.6055138
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6054248, upper bound: 330.6052625
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6057750, upper bound: 330.6055138
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6054248, upper bound: 330.6052625
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6056999, upper bound: 330.6054159
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6053305, upper bound: 330.6051665
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6056998, upper bound: 330.6054159
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6053305, upper bound: 330.6051665
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.78
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -163.2466736, 129.8153534, -163.0936127, 129.6840668, -292.9307251, 292.9089661
1: -136.9022217, 115.2632446, -136.7717133, 115.1303101, -252.0325317, 252.0349579
2: -179.9809570, 116.7970657, -179.8073425, 116.6891479, -296.6700745, 296.6043396
3: -191.1587219, 100.8249054, -190.9800415, 100.7030716, -291.8617859, 291.8049316
4: -175.9459229, 134.8358154, -175.8104553, 134.7310944, -310.6770020, 310.6462708
5: -157.3332825, 122.3597031, -157.1674500, 122.2795105, -279.6127625, 279.5271606
6: -150.8487091, 144.7868042, -150.7566376, 144.6092529, -295.4579468, 295.5433960
7: -163.8679352, 137.5632019, -163.7389374, 137.4696808, -301.3376160, 301.3021240
8: -197.1243286, 134.4198151, -196.9558716, 134.2707977, -331.3951416, 331.3756714
9: -149.1941528, 147.1309357, -149.0731049, 146.9928131, -296.1869507, 296.2039185

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5985639, upper bound: 330.5991089
time: 9.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5966564, upper bound: 330.5965493
time: 8.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -160.9668732, 127.9989243, -163.0602112, 129.6570587, -290.6239319, 291.0591125
1: -134.9642181, 113.6271057, -136.7127228, 115.0717773, -250.0359650, 250.3398285
2: -177.4429626, 115.1576538, -179.7540283, 116.6427994, -294.0857544, 294.9116516
3: -188.4580231, 99.3998184, -190.9226837, 100.6509933, -289.1090088, 290.3224487
4: -173.5004425, 132.9494324, -175.7933502, 134.6785278, -308.1789551, 308.7427368
5: -155.1340027, 120.6530609, -157.1420441, 122.2266922, -277.3606873, 277.7950439
6: -148.7345734, 142.7464905, -150.7257996, 144.5640564, -293.2986145, 293.4722900
7: -161.5670013, 135.6423035, -163.6972351, 137.4364624, -299.0034790, 299.3395386
8: -194.3598785, 132.5309448, -196.9120178, 134.2451019, -328.6049500, 329.4429626
9: -147.1090393, 145.0690765, -149.0469971, 146.9526978, -294.0617371, 294.1160889

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5979850, upper bound: 330.5986576
time: 9.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5960979, upper bound: 330.5961718
time: 7.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -164.4572754, 130.7727356, -166.0113373, 131.9815674, -296.4388428, 296.7840576
1: -137.9124603, 116.1084290, -139.1745911, 117.1350784, -255.0475464, 255.2830200
2: -181.3135376, 117.6564407, -182.9847565, 118.7291183, -300.0426636, 300.6411438
3: -192.5812531, 101.5671082, -194.3661346, 102.4693069, -295.0505676, 295.9332275
4: -177.2482910, 135.8239288, -178.9056702, 137.0776062, -314.3258362, 314.7296143
5: -158.4981537, 123.2606354, -159.9604950, 124.4194489, -282.9176025, 283.2211304
6: -151.9586182, 145.8571777, -153.3795319, 147.1629791, -299.1215820, 299.2366943
7: -165.0847015, 138.5765076, -166.6071625, 139.8718872, -304.9565735, 305.1836548
8: -198.5790253, 135.4004822, -200.4551697, 136.6598206, -335.2388306, 335.8556519
9: -150.2935028, 148.2129669, -151.6839752, 149.5530243, -299.8465271, 299.8969116

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
time: 8.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
time: 8.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -162.1739655, 128.9535217, -165.6411285, 131.6804199, -293.8543701, 294.5946350
1: -135.9715271, 114.4697952, -138.8307190, 116.8322449, -252.8037720, 253.3005066
2: -178.7716980, 116.0145493, -182.5511780, 118.4369888, -297.2086487, 298.5657349
3: -189.8764343, 100.1398697, -193.9021912, 102.2025070, -292.0789490, 294.0420532
4: -174.7990875, 133.9346924, -178.5268707, 136.7488098, -311.5478210, 312.4614258
5: -156.2954712, 121.5513229, -159.6070709, 124.1147995, -280.4102478, 281.1583252
6: -149.8412781, 143.8137665, -153.0396576, 146.8103027, -296.6515808, 296.8533325
7: -162.7801666, 136.6526337, -166.2260590, 139.5544739, -302.3346252, 302.8786926
8: -195.8103638, 133.5087738, -200.0017395, 136.3486176, -332.1589355, 333.5104980
9: -148.2051544, 146.1479645, -151.3513184, 149.2047424, -297.4099121, 297.4992676

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
time: 11.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
time: 7.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -169.5314331, 134.8087006, -162.6898804, 129.3672638, -298.8986816, 297.4985962
1: -142.1986542, 119.7242661, -136.4374237, 114.8498001, -257.0484619, 256.1616821
2: -186.9072571, 121.2403107, -179.3643951, 116.4003448, -303.3076172, 300.6046753
3: -198.6220551, 104.7902298, -190.5119781, 100.4587479, -299.0808105, 295.3021545
4: -182.6681976, 140.0095062, -175.3772888, 134.4023438, -317.0705261, 315.3867798
5: -163.3943787, 127.0285873, -156.7830048, 121.9817963, -285.3761597, 283.8115845
6: -156.6339874, 150.3840942, -150.3845825, 144.2559509, -300.8899536, 300.7686462
7: -170.1195984, 142.8548126, -163.3352814, 137.1312866, -307.2508850, 306.1900635
8: -204.7450409, 139.6323547, -196.4715424, 133.9418030, -338.6868286, 336.1038208
9: -154.9062195, 152.7733154, -148.7065887, 146.6365051, -301.5426636, 301.4798889

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5984295, upper bound: 330.5989740
time: 10.23 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5964395, upper bound: 330.5962699
time: 8.34 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -167.1893768, 132.9455109, -162.6533813, 129.3378143, -296.5271912, 295.5988770
1: -140.2111511, 118.0462189, -136.3758698, 114.7890091, -255.0001373, 254.4220886
2: -184.3017426, 119.5581131, -179.3077087, 116.3519821, -300.6537170, 298.8657532
3: -195.8497772, 103.3251343, -190.4508362, 100.4047318, -296.2545166, 293.7759705
4: -180.1586761, 138.0745850, -175.3566132, 134.3471832, -314.5058594, 313.4311523
5: -161.1368256, 125.2785034, -156.7546082, 121.9265518, -283.0633240, 282.0330200
6: -154.4656677, 148.2898407, -150.3508911, 144.2081146, -298.6737061, 298.6407471
7: -167.7566528, 140.8853760, -163.2902832, 137.0955353, -304.8521118, 304.1756592
8: -201.9079590, 137.6946106, -196.4240265, 133.9136658, -335.8216248, 334.1186218
9: -152.7666168, 150.6602020, -148.6776276, 146.5936890, -299.3602905, 299.3378296

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5978647, upper bound: 330.5985405
time: 8.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5959370, upper bound: 330.5959309
time: 7.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -170.7695923, 135.7886658, -165.5674591, 131.6331787, -302.4027405, 301.3561401
1: -143.2329102, 120.5896606, -138.8068695, 116.8267899, -260.0596008, 259.3965454
2: -188.2707977, 122.1200714, -182.4974976, 118.4113693, -306.6821289, 304.6175537
3: -200.0784760, 105.5500412, -193.8518982, 102.2005310, -302.2789917, 299.4019470
4: -184.0005493, 141.0211487, -178.4296265, 136.7158813, -320.7164307, 319.4507751
5: -164.5868683, 127.9503632, -159.5375519, 124.0915451, -288.6783752, 287.4879150
6: -157.7708282, 151.4792633, -152.9703369, 146.7746582, -304.5454712, 304.4494934
7: -171.3654480, 143.8921967, -166.1631317, 139.4997864, -310.8651428, 310.0553284
8: -206.2335968, 140.6363220, -199.9228516, 136.2982635, -342.5318604, 340.5591736
9: -156.0314484, 153.8809357, -151.2808533, 149.1608887, -305.1923218, 305.1617432

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
time: 8.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
time: 7.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -168.4240723, 133.9227142, -165.1995697, 131.3339081, -299.7579956, 299.1222534
1: -141.2424164, 118.9090576, -138.4650269, 116.5256042, -257.7680054, 257.3740845
2: -185.6613617, 120.4353867, -182.0665131, 118.1211472, -303.7825012, 302.5018921
3: -197.3020020, 104.0828552, -193.3907623, 101.9352188, -299.2372131, 297.4735413
4: -181.4872131, 139.0833130, -178.0532227, 136.3891296, -317.8763428, 317.1365051
5: -162.3259888, 126.1976242, -159.1864319, 123.7886581, -286.1146545, 285.3840332
6: -155.5993347, 149.3819122, -152.6327820, 146.4241333, -302.0234680, 302.0147095
7: -168.9989319, 141.9197540, -165.7843018, 139.1844025, -308.1833191, 307.7040100
8: -203.3923035, 138.6957092, -199.4724274, 135.9888611, -339.3811646, 338.1681519
9: -153.8886566, 151.7646484, -150.9502869, 148.8148499, -302.7034912, 302.7149353

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571
time: 8.19 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571
time: 8.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 17.53 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5985639, upper bound: 330.5991089
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5966564, upper bound: 330.5965493
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5979850, upper bound: 330.5986576
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5960979, upper bound: 330.5961718
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6113917, upper bound: 330.6113135
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6109838, upper bound: 330.6109991
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5984295, upper bound: 330.5989740
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5964395, upper bound: 330.5962699
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5978647, upper bound: 330.5985405
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.5959370, upper bound: 330.5959309
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6112372, upper bound: 330.6111564
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 9, lower bound: -330.6108571, upper bound: 330.6108571

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -160.9811096, 128.0377655, -166.0113373, 131.9815674, -292.9626770, 294.0491028
1: -135.0431824, 113.7147980, -139.1745911, 117.1350784, -252.1782532, 252.8893890
2: -177.5227203, 115.2235107, -182.9847565, 118.7291183, -296.2518311, 298.2082214
3: -188.5384979, 99.4641953, -194.3661346, 102.4693069, -291.0078125, 293.8303223
4: -173.5497894, 133.0251312, -178.9056702, 137.0776062, -310.6273499, 311.9307861
5: -155.1680756, 120.7088470, -159.9604950, 124.4194489, -279.5874939, 280.6693420
6: -148.8241577, 142.8096313, -153.3795319, 147.1629791, -295.9870911, 296.1891479
7: -161.6533203, 135.7086792, -166.6071625, 139.8718872, -301.5251465, 302.3158569
8: -194.4116821, 132.5667267, -200.4551697, 136.6598206, -331.0714722, 333.0219116
9: -147.1744385, 145.1539764, -151.6839752, 149.5530243, -296.7274780, 296.8379517

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
time: 8.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6051147, upper bound: 330.6049178
time: 8.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -163.6821594, 130.1603241, -166.0113373, 131.9815674, -295.6637268, 296.1716614
1: -137.2642670, 115.5665131, -139.1745911, 117.1350784, -254.3993530, 254.7411041
2: -180.4598846, 117.1070862, -182.9847565, 118.7291183, -299.1889954, 300.0917969
3: -191.6704559, 101.0941467, -194.3661346, 102.4693069, -294.1397705, 295.4602661
4: -176.4134674, 135.1909790, -178.9056702, 137.0776062, -313.4909668, 314.0966492
5: -157.7528076, 122.6852112, -159.9604950, 124.4194489, -282.1722412, 282.6456909
6: -151.2475891, 145.1702576, -153.3795319, 147.1629791, -298.4105530, 298.5498047
7: -164.3062592, 137.9278107, -166.6071625, 139.8718872, -304.1781006, 304.5349121
8: -197.6480103, 134.7714539, -200.4551697, 136.6598206, -334.3078003, 335.2265930
9: -149.5892181, 147.5190125, -151.6839752, 149.5530243, -299.1422424, 299.2029114

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
time: 9.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6051147, upper bound: 330.6049178
time: 8.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -158.8203735, 126.3182755, -165.6411285, 131.6804199, -290.5006714, 291.9593506
1: -133.2059326, 112.1659851, -138.8307190, 116.8322449, -250.0381775, 250.9967041
2: -175.1200409, 113.6713791, -182.5511780, 118.4369888, -293.5570374, 296.2225647
3: -185.9826508, 98.1160202, -193.9021912, 102.2025070, -288.1851501, 292.0182190
4: -171.2336273, 131.2359924, -178.5268707, 136.7488098, -307.9824219, 309.7627869
5: -153.0866852, 119.0905151, -159.6070709, 124.1147995, -277.2014465, 278.6974792
6: -146.8201904, 140.8791351, -153.0396576, 146.8103027, -293.6304932, 293.9187012
7: -159.4723206, 133.8888702, -166.2260590, 139.5544739, -299.0267639, 300.1149292
8: -191.7920685, 130.7786102, -200.0017395, 136.3486176, -328.1406250, 330.7803345
9: -145.1976166, 143.2012787, -151.3513184, 149.2047424, -294.4023438, 294.5526123

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6060166, upper bound: 330.6059880
time: 8.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6063291, upper bound: 330.6063515
time: 8.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -161.3977051, 128.3401642, -165.6411285, 131.6804199, -293.0781250, 293.9812622
1: -135.3224030, 113.9271240, -138.8307190, 116.8322449, -252.1546478, 252.7578430
2: -177.9167786, 115.4644012, -182.5511780, 118.4369888, -296.3537598, 298.0155640
3: -188.9644012, 99.6662369, -193.9021912, 102.2025070, -291.1669006, 293.5684204
4: -173.9630127, 133.3007660, -178.5268707, 136.7488098, -310.7118225, 311.8275452
5: -155.5490417, 120.9749985, -159.6070709, 124.1147995, -279.6638184, 280.5820007
6: -149.1292725, 143.1258698, -153.0396576, 146.8103027, -295.9395752, 296.1654968
7: -162.0006561, 136.0029602, -166.2260590, 139.5544739, -301.5551147, 302.2290039
8: -194.8780518, 132.8788452, -200.0017395, 136.3486176, -331.2266846, 332.8805847
9: -147.4998779, 145.4530640, -151.3513184, 149.2047424, -296.7046204, 296.8043518

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6060166, upper bound: 330.6059880
time: 6.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6063291, upper bound: 330.6063515
time: 9.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -167.0283661, 132.8393555, -165.5674591, 131.6331787, -298.6614990, 298.4067993
1: -140.1380005, 118.0071106, -138.8068695, 116.8267899, -256.9647827, 256.8139648
2: -184.1849213, 119.4983215, -182.4974976, 118.4113693, -302.5962524, 301.9958191
3: -195.7153320, 103.2801514, -193.8518982, 102.2005310, -297.9158630, 297.1320496
4: -180.0158997, 137.9974060, -178.4296265, 136.7158813, -316.7317810, 316.4269714
5: -160.9952087, 125.2046814, -159.5375519, 124.0915451, -285.0867310, 284.7422485
6: -154.3877716, 148.1943054, -152.9703369, 146.7746582, -301.1624146, 301.1646423
7: -167.6669312, 140.8003082, -166.1631317, 139.4997864, -307.1666870, 306.9634399
8: -201.7470093, 137.5783081, -199.9228516, 136.2982635, -338.0452881, 337.5011597
9: -152.6701355, 150.5831451, -151.2808533, 149.1608887, -301.8309326, 301.8639526

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
time: 8.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6050979, upper bound: 330.6049125
time: 9.27 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -169.9632111, 135.1506805, -165.5674591, 131.6331787, -301.5963745, 300.7181396
1: -142.5575104, 120.0251083, -138.8068695, 116.8267899, -259.3843079, 258.8319702
2: -187.3819122, 121.5475693, -182.4974976, 118.4113693, -305.7932739, 304.0450745
3: -199.1299286, 105.0571671, -193.8518982, 102.2005310, -301.3304443, 298.9090576
4: -183.1317291, 140.3615265, -178.4296265, 136.7158813, -319.8475952, 318.7911377
5: -163.8101654, 127.3513260, -159.5375519, 124.0915451, -287.9016113, 286.8888855
6: -157.0293732, 150.7642059, -152.9703369, 146.7746582, -303.8040161, 303.7344971
7: -170.5540771, 143.2162170, -166.1631317, 139.4997864, -310.0538635, 309.3793335
8: -205.2648315, 139.9809570, -199.9228516, 136.2982635, -341.5631104, 339.9038086
9: -155.2978973, 153.1582184, -151.2808533, 149.1608887, -304.4587402, 304.4390259

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
time: 8.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6050979, upper bound: 330.6049126
time: 7.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -164.7942657, 131.0619659, -165.1995697, 131.3339081, -296.1281738, 296.2615356
1: -138.2399139, 116.4053497, -138.4650269, 116.5256042, -254.7655182, 254.8703613
2: -181.7002106, 117.8936691, -182.0665131, 118.1211472, -299.8213501, 299.9601746
3: -193.0719757, 101.8828888, -193.3907623, 101.9352188, -295.0072021, 295.2735901
4: -177.6234436, 136.1482544, -178.0532227, 136.3891296, -314.0125732, 314.2014465
5: -158.8426819, 123.5333481, -159.1864319, 123.7886581, -282.6313477, 282.7197266
6: -152.3169250, 146.1986084, -152.6327820, 146.4241333, -298.7410278, 298.8313904
7: -165.4131775, 138.9189606, -165.7843018, 139.1844025, -304.5975037, 304.7032166
8: -199.0368652, 135.7298584, -199.4724274, 135.9888611, -335.0256958, 335.2022705
9: -150.6284943, 148.5656586, -150.9502869, 148.8148499, -299.4432983, 299.5159302

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6058894, upper bound: 330.6058367
time: 8.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062134, upper bound: 330.6062134
time: 9.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -167.6163788, 133.2836761, -165.1995697, 131.3339081, -298.9502869, 298.4831848
1: -140.5659637, 118.3436279, -138.4650269, 116.5256042, -257.0915527, 256.8086548
2: -184.7710724, 119.8619919, -182.0665131, 118.1211472, -302.8922119, 301.9284973
3: -196.3519135, 103.5891647, -193.3907623, 101.9352188, -298.2871094, 296.9798584
4: -180.6170502, 138.4226227, -178.0532227, 136.3891296, -317.0061646, 316.4757996
5: -161.5479431, 125.5975723, -159.1864319, 123.7886581, -285.3366089, 284.7839355
6: -154.8567047, 148.6657104, -152.6327820, 146.4241333, -301.2807617, 301.2984924
7: -168.1862793, 141.2426758, -165.7843018, 139.1844025, -307.3705750, 307.0269470
8: -202.4220428, 138.0393524, -199.4724274, 135.9888611, -338.4108276, 337.5117798
9: -153.1539459, 151.0407867, -150.9502869, 148.8148499, -301.9688110, 301.9910889

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6002369, upper bound: 330.6058367
time: 9.32 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062134, upper bound: 330.6062134
time: 6.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 17.22 seconds
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6051147, upper bound: 330.6049178
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6051147, upper bound: 330.6049178
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6060166, upper bound: 330.6059880
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6063291, upper bound: 330.6063515
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6060166, upper bound: 330.6059880
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6063291, upper bound: 330.6063515
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6050979, upper bound: 330.6049125
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6062077, upper bound: 330.6057881
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6050979, upper bound: 330.6049126
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6058894, upper bound: 330.6058367
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6062134, upper bound: 330.6062134
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6002369, upper bound: 330.6058367
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 17.22
Output dim: 9, lower bound: -330.6062134, upper bound: 330.6062134
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=333.2668151855469
rel_dist={9: [-330.62201561424183, 330.62201560609844]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215173, upper bound: 330.6215895
time: 7.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6214242, upper bound: 330.6214242
time: 8.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.69
Output dim: 9, lower bound: -330.6215173, upper bound: 330.6215895
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.69
Output dim: 9, lower bound: -330.6214242, upper bound: 330.6214242

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -180.3420868, 143.4214325, -182.9472198, 145.4716492, -325.8136902, 326.3686218
1: -151.3821564, 127.4255600, -153.5455780, 129.2407532, -280.6228027, 280.9711304
2: -198.9765472, 128.9956970, -201.8273315, 130.8524628, -329.8290100, 330.8230286
3: -211.3465729, 111.4465866, -214.3889618, 113.0435562, -324.3901367, 325.8354797
4: -194.3313751, 148.9740448, -197.1164246, 151.0996094, -345.4309387, 346.0904541
5: -173.8146667, 135.1881104, -176.3052521, 137.1186371, -310.9332886, 311.4933472
6: -166.6844940, 159.9958649, -169.0819550, 162.2864227, -328.9708557, 329.0777588
7: -181.1339569, 151.9680023, -183.7369537, 154.1575775, -335.2915344, 335.7049561
8: -217.8480682, 148.5378723, -220.9689026, 150.6581879, -368.5062561, 369.5067444
9: -164.8039093, 162.5196381, -167.1781006, 164.8440247, -329.6479492, 329.6976929

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215153, upper bound: 330.6215788
time: 8.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6215131, upper bound: 330.6215788
time: 7.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -186.7120209, 148.4876099, -182.4551239, 145.0851135, -331.7971191, 330.9427490
1: -156.7530975, 131.9496307, -153.1375885, 128.8986816, -285.6517639, 285.0872192
2: -206.0017548, 133.5013428, -201.2879028, 130.5008545, -336.5026245, 334.7892456
3: -218.9201202, 115.4685135, -213.8171844, 112.7444000, -331.6643982, 329.2856750
4: -201.1480560, 154.2209930, -196.5892029, 150.6985321, -351.8465881, 350.8100891
5: -179.9620514, 139.9226074, -175.8359833, 136.7547150, -316.7167358, 315.7586060
6: -172.5550537, 165.6704559, -168.6287689, 161.8552856, -334.4103394, 334.2991943
7: -187.4775543, 157.3350067, -183.2448425, 153.7446747, -341.2222290, 340.5798340
8: -225.5724030, 153.8252563, -220.3789825, 150.2577362, -375.8301392, 374.2042236
9: -170.5974579, 168.2449493, -166.7305603, 164.4079895, -335.0054321, 334.9754333

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6214242, upper bound: 330.6214206
time: 7.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6214206, upper bound: 330.6214206
time: 11.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.22
Output dim: 9, lower bound: -330.6215153, upper bound: 330.6215788
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.22
Output dim: 9, lower bound: -330.6215131, upper bound: 330.6215788
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.22
Output dim: 9, lower bound: -330.6214242, upper bound: 330.6214206
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.22
Output dim: 9, lower bound: -330.6214206, upper bound: 330.6214206

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -179.3062286, 142.6018066, -178.9070740, 142.2861481, -321.5923767, 321.5088806
1: -150.5170135, 126.7018738, -150.1988525, 126.4462814, -276.9632874, 276.9007263
2: -197.8359528, 128.2598877, -197.4095154, 128.0173645, -325.8533020, 325.6694031
3: -210.1287994, 110.8114777, -209.6794739, 110.5915756, -320.7203674, 320.4909058
4: -193.2166901, 148.1278229, -192.8101807, 147.8323975, -341.0490417, 340.9379272
5: -172.8171234, 134.4174957, -172.4314423, 134.1432648, -306.9603882, 306.8489380
6: -165.7336426, 159.0795593, -165.4237518, 158.7375031, -324.4711304, 324.5032349
7: -180.0921478, 151.1004181, -179.7342377, 150.8119354, -330.9040833, 330.8345642
8: -216.6033630, 147.6977692, -216.1187744, 147.3612518, -363.9645081, 363.8165283
9: -163.8627014, 161.5932159, -163.5439606, 161.2760315, -325.1387329, 325.1371765

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
time: 8.13 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
time: 6.93 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -180.3420868, 143.4214325, -181.9926910, 144.7169800, -325.0590820, 325.4140015
1: -151.3821564, 127.4255600, -152.7468109, 128.5731201, -279.9552612, 280.1723633
2: -198.9765472, 128.9956970, -200.7756958, 130.1753693, -329.1518555, 329.7713928
3: -211.3465729, 111.4465866, -213.2669220, 112.4605637, -323.8071289, 324.7135010
4: -194.3313751, 148.9740448, -196.0881195, 150.3194580, -344.6508179, 345.0621643
5: -173.8146667, 135.1881104, -175.3864594, 136.4099274, -310.2246094, 310.5745544
6: -166.6844940, 159.9958649, -168.2052917, 161.4403992, -328.1248169, 328.2011108
7: -181.1339569, 151.9680023, -182.7774200, 153.3581390, -334.4920959, 334.7454224
8: -217.8480682, 148.5378723, -219.8223572, 149.8828430, -367.7308655, 368.3601685
9: -164.8039093, 162.5196381, -166.3103790, 163.9891510, -328.7930298, 328.8299866

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
time: 8.57 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164244
time: 7.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -185.6682892, 147.6616058, -178.4435425, 141.9219971, -327.5902710, 326.1051025
1: -155.8811340, 131.2201996, -149.8146515, 126.1238327, -282.0049438, 281.0348206
2: -204.8523712, 132.7597504, -196.9015350, 127.6863556, -332.5387268, 329.6611938
3: -217.6925201, 114.8283463, -209.1402740, 110.3096695, -328.0021973, 323.9685364
4: -200.0248260, 153.3681335, -192.3133698, 147.4546814, -347.4794617, 345.6813965
5: -178.9567719, 139.1459045, -171.9895630, 133.8007202, -312.7574463, 311.1354675
6: -171.5966949, 164.7471008, -164.9968414, 158.3312378, -329.9279175, 329.7439575
7: -186.4275818, 156.4606476, -179.2707367, 150.4230194, -336.8505859, 335.7313232
8: -224.3176575, 152.9785919, -215.5627441, 146.9838715, -371.3015137, 368.5413208
9: -169.6488647, 167.3111572, -163.1223450, 160.8652344, -330.5140076, 330.4335022

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117298, upper bound: 330.6114486
time: 9.13 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6162817, upper bound: 330.6162817
time: 7.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -186.7120209, 148.4876099, -181.5032349, 144.3325500, -331.0445557, 329.9908447
1: -156.7530975, 131.9496307, -152.3410492, 128.2328796, -284.9859619, 284.2906494
2: -206.0017548, 133.5013428, -200.2391968, 129.8256073, -335.8273621, 333.7405396
3: -218.9201202, 115.4685135, -212.6982727, 112.1629868, -331.0830994, 328.1667786
4: -201.1480560, 154.2209930, -195.5637512, 149.9205322, -351.0685730, 349.7846680
5: -179.9620514, 139.9226074, -174.9197845, 136.0479279, -316.0099487, 314.8423462
6: -172.5550537, 165.6704559, -167.7545624, 161.0115814, -333.5666504, 333.4250183
7: -187.4775543, 157.3350067, -182.2879944, 152.9474335, -340.4249878, 339.6229858
8: -225.5724030, 153.8252563, -219.2356720, 149.4844818, -375.0568848, 373.0609131
9: -170.5974579, 168.2449493, -165.8651276, 163.5554352, -334.1528320, 334.1100464

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117298, upper bound: 330.6114486
time: 8.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6162817, upper bound: 330.6162817
time: 9.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.35 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164244
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6117298, upper bound: 330.6114486
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6162817, upper bound: 330.6162817
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6117298, upper bound: 330.6114486
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.35
Output dim: 9, lower bound: -330.6162817, upper bound: 330.6162817

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -159.9355774, 127.1944962, -172.7835388, 137.4119568, -297.3475037, 299.9779968
1: -134.1224518, 112.9295883, -145.0061951, 122.0863647, -256.2088013, 257.9357910
2: -176.2944946, 114.4742126, -190.5930481, 123.6558838, -299.9503479, 305.0672607
3: -187.1986084, 98.7907104, -202.4250946, 106.7867584, -293.9853516, 301.2157593
4: -172.3615723, 132.0894318, -186.2186737, 142.7562714, -315.1178589, 318.3081055
5: -154.1713867, 119.8309402, -166.5335236, 129.5276947, -283.6990967, 286.3644409
6: -147.7380371, 141.8783112, -159.7279816, 153.2970886, -301.0350952, 301.6062927
7: -160.5012054, 134.7406921, -173.5364990, 145.6351318, -306.1363525, 308.2771912
8: -193.0723877, 131.6917114, -208.6737518, 142.2994995, -335.3718872, 340.3654480
9: -146.1438904, 144.1676331, -157.9397430, 155.7591705, -301.9030762, 302.1073608

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
time: 8.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
time: 8.69 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -167.5298767, 133.2292480, -176.1045990, 140.0563507, -307.5861816, 309.3338623
1: -140.5399780, 118.3326874, -147.8245239, 124.4556122, -264.9956055, 266.1571960
2: -184.7501526, 119.8682556, -194.2962799, 126.0208206, -310.7709656, 314.1645203
3: -196.2275696, 103.5082855, -206.3710480, 108.8541641, -305.0817261, 309.8793335
4: -180.5502472, 138.3833160, -189.7971039, 145.5134125, -326.0636597, 328.1804199
5: -161.4697266, 125.5695343, -169.7306061, 132.0373383, -293.5070496, 295.3001099
6: -154.8194580, 148.6204529, -162.8272400, 156.2487030, -311.0681152, 311.4476929
7: -168.2040863, 141.1760559, -176.9052734, 148.4506226, -316.6546631, 318.0812073
8: -202.3098297, 137.9613953, -212.7178650, 145.0448456, -347.3546753, 350.6792603
9: -153.1157684, 151.0040283, -160.9867859, 158.7563782, -311.8721313, 311.9907532

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6164244
time: 9.24 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6164245
time: 9.43 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -160.9380188, 127.9871902, -175.9278717, 139.8905182, -300.8285522, 303.9150696
1: -134.9591827, 113.6294250, -147.6053619, 124.2554779, -259.2146606, 261.2348022
2: -177.3978271, 115.1857910, -194.0265350, 125.8562851, -303.2540894, 309.2123108
3: -188.3767700, 99.4048767, -206.0838928, 108.6947403, -297.0714722, 305.4887695
4: -173.4401855, 132.9075317, -189.5606689, 145.2923889, -318.7325439, 322.4682007
5: -155.1359100, 120.5766220, -169.5451050, 131.8406830, -286.9765625, 290.1217346
6: -148.6571503, 142.7648010, -162.5660095, 156.0528107, -304.7099609, 305.3308105
7: -161.5084839, 135.5796204, -176.6389771, 148.2330322, -309.7414856, 312.2185974
8: -194.2768555, 132.5039368, -212.4504089, 144.8707123, -339.1475220, 344.9543152
9: -147.0541840, 145.0638123, -160.7614288, 158.5274506, -305.5815735, 305.8251648

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
time: 8.48 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115183
time: 8.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -179.1150818, 142.4260406, -310.9701843, 313.1463013
1: -141.3865662, 119.0408478, -150.3081665, 126.5263977, -267.9129639, 269.3489685
2: -185.8666382, 120.5881119, -197.5764313, 128.1248474, -313.9914856, 318.1645508
3: -197.4193420, 104.1297684, -209.8672943, 110.6750793, -308.0944214, 313.9970398
4: -181.6414948, 139.2112579, -192.9927521, 147.9371185, -329.5786133, 332.2040100
5: -162.4455872, 126.3240967, -172.6125031, 134.2467804, -296.6923218, 298.9365845
6: -155.7493744, 149.5174408, -165.5366211, 158.8834991, -314.6328735, 315.0539551
7: -169.2233276, 142.0249481, -179.8708038, 150.9317017, -320.1550293, 321.8956909
8: -203.5285492, 138.7832489, -216.3275299, 147.5029449, -351.0314941, 355.1106567
9: -154.0368805, 151.9107056, -163.6829376, 161.3998566, -315.4367371, 315.5935974

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164244
time: 8.36 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164244
time: 7.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -166.0497131, 132.0517883, -172.3184052, 137.0466309, -303.0963440, 304.3701782
1: -139.2739410, 117.2724991, -144.6206665, 121.7628479, -261.0367737, 261.8931580
2: -183.0336304, 118.7921906, -190.0832825, 123.3238068, -306.3574219, 308.8754578
3: -194.4714661, 102.6496353, -201.8840637, 106.5039291, -300.9754028, 304.5336304
4: -178.9068604, 137.1209106, -185.7201843, 142.3773346, -321.2841797, 322.8410339
5: -160.0654755, 124.3748932, -166.0902100, 129.1840363, -289.2495117, 290.4650879
6: -153.3670502, 147.3245544, -159.2995911, 152.8894653, -306.2564392, 306.6240845
7: -166.5834351, 139.8889160, -173.0713654, 145.2449341, -311.8283691, 312.9602661
8: -200.4835205, 136.7626190, -208.1158447, 141.9208679, -342.4043884, 344.8784790
9: -151.7048798, 149.6608887, -157.5167542, 155.3470459, -307.0519104, 307.1776428

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6061939, upper bound: 330.6058907
time: 8.02 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6056968, upper bound: 330.6055528
time: 8.16 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -173.7886353, 138.2027893, -175.6406250, 139.6918335, -313.4804688, 313.8434143
1: -145.8163757, 122.7761993, -147.4399261, 124.1328278, -269.9491882, 270.2161255
2: -191.6482391, 124.2939758, -193.7877502, 125.6894684, -317.3376770, 318.0817261
3: -203.6609039, 107.4569702, -205.8312988, 108.5719833, -312.2328491, 313.2882690
4: -187.2452087, 143.5374451, -189.2997894, 145.1354065, -332.3806152, 332.8372192
5: -167.5066223, 130.2201843, -169.2882690, 131.6944580, -299.2010803, 299.5084534
6: -160.5823059, 154.1951141, -162.3999329, 155.8420105, -316.4243164, 316.5950317
7: -174.4300385, 146.4473724, -176.4412994, 148.0612946, -322.4913330, 322.8886414
8: -209.9001770, 143.1529236, -212.1613464, 144.6671295, -354.5672913, 355.3142395
9: -158.8049164, 156.6249695, -160.5647430, 158.3452301, -317.1501465, 317.1896667

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
time: 8.26 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541
time: 7.78 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -167.0620422, 132.8526154, -175.4369812, 139.5049591, -306.5670166, 308.2895813
1: -140.1192017, 117.9796524, -147.1983337, 123.9142303, -264.0334473, 265.1779785
2: -184.1480865, 119.5109940, -193.4883728, 125.5055313, -309.6535645, 312.9993591
3: -195.6618958, 103.2701416, -205.5134735, 108.3963089, -304.0581665, 308.7836304
4: -179.9962006, 137.9474030, -189.0346222, 144.8923035, -324.8884888, 326.9820251
5: -161.0397797, 125.1281433, -169.0771027, 131.4776154, -292.5173340, 294.2052307
6: -154.2957306, 148.2198486, -162.1138458, 155.6227112, -309.9184570, 310.3336792
7: -167.6010132, 140.7364502, -176.1480103, 147.8210907, -315.4219971, 316.8844299
8: -201.7003937, 137.5831451, -211.8618469, 144.4711761, -346.1715698, 349.4449768
9: -152.6244354, 150.5662994, -160.3148804, 158.0924377, -310.7168579, 310.8811646

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6061939, upper bound: 330.6058907
time: 7.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6056968, upper bound: 330.6055528
time: 7.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -174.8271179, 139.0247040, -178.6249847, 142.0411377, -316.8682556, 317.6496887
1: -146.6840363, 123.5020981, -149.9017944, 126.1857071, -272.8697510, 273.4039001
2: -192.7919464, 125.0317993, -197.0391998, 127.7745895, -320.5664673, 322.0709839
3: -204.8824310, 108.0939713, -209.2978363, 110.3771210, -315.2595520, 317.3918152
4: -188.3628540, 144.3860474, -192.4676514, 147.5376282, -335.9004517, 336.8536987
5: -168.5068359, 130.9931030, -172.1452026, 133.8842468, -302.3910522, 303.1382446
6: -161.5359650, 155.1138611, -165.0852356, 158.4541016, -319.9900513, 320.1990967
7: -175.4748840, 147.3174591, -179.3806305, 150.5204315, -325.9952698, 326.6980896
8: -211.1485291, 143.9952850, -215.7399750, 147.1040955, -358.2526245, 359.7351990
9: -159.7488251, 157.5541534, -163.2371216, 160.9655457, -320.7143250, 320.7912598

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
time: 8.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109540
time: 8.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6164244
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6164245
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115183
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164244
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164244
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6061939, upper bound: 330.6058907
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6056968, upper bound: 330.6055528
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6061939, upper bound: 330.6058907
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6056968, upper bound: 330.6055528
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109540

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -159.9355774, 127.1944962, -170.1647034, 135.3504181, -295.2860107, 297.3591919
1: -134.1224518, 112.9295883, -142.8311920, 120.2607574, -254.3832092, 255.7607574
2: -176.2944946, 114.4742126, -187.7271423, 121.7887802, -298.0832214, 302.2013550
3: -187.1986084, 98.7907104, -199.3649750, 105.1810455, -292.3796387, 298.1557007
4: -172.3615723, 132.0894318, -183.4190826, 140.6189117, -312.9804688, 315.5084839
5: -154.1713867, 119.8309402, -164.0296936, 127.5869675, -281.7583618, 283.8606262
6: -147.7380371, 141.8783112, -157.3169861, 150.9943085, -298.7323303, 299.1953125
7: -160.5012054, 134.7406921, -170.9190826, 143.4337921, -303.9349976, 305.6597595
8: -193.0723877, 131.6917114, -205.5358124, 140.1677399, -333.2401123, 337.2275085
9: -146.1438904, 144.1676331, -155.5528717, 153.4216919, -299.5655823, 299.7204590

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062120, upper bound: 330.6062469
time: 8.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6058197, upper bound: 330.6056798
time: 8.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -159.9355774, 127.1944962, -176.2048798, 140.1508484, -300.0863953, 303.3993530
1: -134.1224518, 112.9295883, -147.9218750, 124.5494614, -258.6719055, 260.8514709
2: -176.2944946, 114.4742126, -194.3873596, 126.0602264, -302.3546753, 308.8615723
3: -187.1986084, 98.7907104, -206.5396729, 108.9942703, -296.1928711, 305.3303833
4: -172.3615723, 132.0894318, -189.8813629, 145.5872650, -317.9488220, 321.9707642
5: -154.1713867, 119.8309402, -169.8521118, 132.0785522, -286.2499390, 289.6829834
6: -147.7380371, 141.8783112, -162.8774414, 156.3737030, -304.1117249, 304.7557373
7: -160.5012054, 134.7406921, -176.9296265, 148.5225220, -309.0237427, 311.6703186
8: -193.0723877, 131.6917114, -212.8667297, 145.1739349, -338.2463379, 344.5584412
9: -146.1438904, 144.1676331, -161.0461273, 158.8487244, -304.9925842, 305.2137451

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062120, upper bound: 330.6062469
time: 7.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6058197, upper bound: 330.6056798
time: 8.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -167.5298767, 133.2292480, -173.4898071, 137.9979706, -305.5278320, 306.7190552
1: -140.5399780, 118.3326874, -145.6527710, 122.6327515, -263.1727295, 263.9854736
2: -184.7501526, 119.8682556, -191.4347382, 124.1565933, -308.9067383, 311.3029785
3: -196.2275696, 103.5082855, -203.3154755, 107.2506943, -303.4782715, 306.8237610
4: -180.5502472, 138.3833160, -187.0016022, 143.3792114, -323.9294128, 325.3849182
5: -161.4697266, 125.5695343, -167.2305298, 130.0995026, -291.5692139, 292.8000488
6: -154.8194580, 148.6204529, -160.4200287, 153.9492645, -308.7687073, 309.0404663
7: -168.2040863, 141.1760559, -174.2919312, 146.2527466, -314.4567871, 315.4678650
8: -202.3098297, 137.9613953, -209.5846100, 142.9162598, -345.2260742, 347.5460205
9: -153.1157684, 151.0040283, -158.6035156, 156.4223480, -309.5381165, 309.6075439

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114898, upper bound: 330.6116002
time: 7.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6111270, upper bound: 330.6111304
time: 7.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -167.5298767, 133.2292480, -179.5478516, 142.8125458, -310.3423767, 312.7770996
1: -140.5399780, 118.3326874, -150.7603149, 126.9350281, -267.4750061, 269.0929871
2: -184.7501526, 119.8682556, -198.1141357, 128.4421997, -313.1923523, 317.9823914
3: -196.2275696, 103.5082855, -210.5097809, 111.0752335, -307.3027954, 314.0180664
4: -180.5502472, 138.3833160, -193.4830322, 148.3643951, -328.9146423, 331.8662720
5: -161.4697266, 125.5695343, -173.0708008, 134.6059265, -296.0756531, 298.6403198
6: -154.8194580, 148.6204529, -165.9969788, 159.3465729, -314.1659546, 314.6174316
7: -168.2040863, 141.1760559, -180.3202820, 151.3572540, -319.5613098, 321.4962769
8: -202.3098297, 137.9613953, -216.9383545, 147.9387665, -350.2485352, 354.8997192
9: -153.1157684, 151.0040283, -164.1131134, 161.8661499, -314.9819336, 315.1171265

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114898, upper bound: 330.6116002
time: 8.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6111270, upper bound: 330.6111304
time: 8.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -160.9380188, 127.9871902, -173.3167419, 137.8356323, -298.7736206, 301.3039246
1: -134.9591827, 113.6294250, -145.4371490, 122.4361725, -257.3953552, 259.0665894
2: -177.3978271, 115.1857910, -191.1693268, 123.9951248, -301.3929138, 306.3551025
3: -188.3767700, 99.4048767, -203.0345764, 107.0943069, -295.4710388, 302.4394531
4: -173.4401855, 132.9075317, -186.7693634, 143.1620636, -316.6022034, 319.6768799
5: -155.1359100, 120.5766220, -167.0489807, 129.9056549, -285.0415649, 287.6255493
6: -148.6571503, 142.7648010, -160.1628265, 153.7571259, -302.4142761, 302.9275818
7: -161.5084839, 135.5796204, -174.0298615, 146.0383759, -307.5468445, 309.6094971
8: -194.2768555, 132.5039368, -209.3224182, 142.7457581, -337.0225830, 341.8262634
9: -147.0541840, 145.0638123, -158.3818665, 156.1977539, -303.2519531, 303.4456177

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115183
time: 9.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
time: 8.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -160.9380188, 127.9871902, -179.5378876, 142.7784882, -303.7164917, 307.5250854
1: -134.9591827, 113.6294250, -150.6759186, 126.8520966, -261.8112793, 264.3053589
2: -177.3978271, 115.1857910, -198.0240936, 128.3885803, -305.7864075, 313.2098389
3: -188.3767700, 99.4048767, -210.4280396, 111.0190048, -299.3957825, 309.8328857
4: -173.4401855, 132.9075317, -193.4234619, 148.2806091, -321.7207642, 326.3309937
5: -155.1359100, 120.5766220, -173.0454254, 134.5266724, -289.6625061, 293.6220093
6: -148.6571503, 142.7648010, -165.8876343, 159.2951660, -307.9523315, 308.6524048
7: -161.5084839, 135.5796204, -180.2161713, 151.2738495, -312.7823486, 315.7957764
8: -194.2768555, 132.5039368, -216.8638916, 147.9069824, -342.1837158, 349.3677979
9: -147.0541840, 145.0638123, -164.0352478, 161.7828217, -308.8370056, 309.0989990

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
time: 8.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115183
time: 7.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -176.5046692, 140.3715973, -308.9157410, 310.5358887
1: -141.3865662, 119.0408478, -148.1404572, 124.7074661, -266.0940247, 267.1812439
2: -185.8666382, 120.5881119, -194.7199707, 126.2643127, -312.1308899, 315.3080444
3: -197.4193420, 104.1297684, -206.8185730, 109.0748520, -306.4941711, 310.9483032
4: -181.6414948, 139.2112579, -190.2020264, 145.8072510, -327.4487305, 329.4132690
5: -162.4455872, 126.3240967, -170.1168823, 132.3124084, -294.7579956, 296.4409790
6: -155.7493744, 149.5174408, -163.1342163, 156.5882874, -312.3376465, 312.6516113
7: -169.2233276, 142.0249481, -177.2624359, 148.7376556, -317.9609985, 319.2872925
8: -203.5285492, 138.7832489, -213.2003174, 145.3784485, -348.9069519, 351.9834900
9: -154.0368805, 151.9107056, -161.3039856, 159.0707397, -313.1076050, 313.2146912

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
time: 6.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
time: 7.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -168.5441437, 134.0312347, -182.7472534, 145.3333740, -313.8775024, 316.7784424
1: -141.3865662, 119.0408478, -153.4015808, 129.1404419, -270.5270081, 272.4423828
2: -185.8666382, 120.5881119, -201.6020966, 130.6759338, -316.5425110, 322.1901855
3: -197.4193420, 104.1297684, -214.2376709, 113.0151443, -310.4344788, 318.3674316
4: -181.6414948, 139.2112579, -196.8806305, 150.9459534, -332.5874634, 336.0918274
5: -162.4455872, 126.3240967, -176.1361847, 136.9507599, -299.3963013, 302.4602661
6: -155.7493744, 149.5174408, -168.8808594, 162.1490784, -317.8984375, 318.3982849
7: -169.2233276, 142.0249481, -183.4717407, 153.9940948, -323.2174072, 325.4966431
8: -203.5285492, 138.7832489, -220.7734528, 150.5581360, -354.0866699, 359.5566711
9: -154.0368805, 151.9107056, -166.9790344, 164.6789398, -318.7158203, 318.8896484

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
time: 6.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
time: 8.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -171.5014038, 136.3778381, -164.7333527, 130.9892883, -302.4906921, 301.1112061
1: -143.8695526, 121.1325226, -138.1609344, 116.2955093, -260.1650391, 259.2934265
2: -189.0983124, 122.6517563, -181.6279297, 117.8570175, -306.9552917, 304.2796936
3: -200.9517975, 106.0213394, -192.9180450, 101.7225113, -302.6742859, 298.9393921
4: -184.7860718, 141.6393890, -177.5726929, 136.0875702, -320.8735657, 319.2120361
5: -165.2965393, 128.5040894, -158.7485352, 123.5122299, -288.8087158, 287.2526245
6: -158.4588013, 152.1450958, -152.2752380, 146.0663300, -304.5250549, 304.4203491
7: -172.1127625, 144.5157166, -165.3936768, 138.8517609, -310.9645386, 309.9093933
8: -207.1282654, 141.2577667, -198.9449310, 135.6259918, -342.7542419, 340.2026672
9: -156.7084656, 154.5532379, -150.5709991, 148.4691620, -305.1776123, 305.1242371

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5991811, upper bound: 330.5998955
time: 7.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5966141, upper bound: 330.5963964
time: 8.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -169.2241211, 134.5648956, -164.7496338, 131.0025635, -300.2266235, 299.3145142
1: -141.9356079, 119.4997559, -138.1452179, 116.2733459, -258.2089539, 257.6449280
2: -186.5636597, 121.0161514, -181.6303711, 117.8460236, -304.4096680, 302.6465149
3: -198.2561951, 104.5955200, -192.9185638, 101.7006149, -299.9568176, 297.5140686
4: -182.3441315, 139.7560883, -177.6086121, 136.0771179, -318.4212341, 317.3646240
5: -163.1000366, 126.8008728, -158.7714844, 123.4974594, -286.5975037, 285.5723267
6: -156.3492126, 150.1076813, -152.2914276, 146.0659180, -302.4151306, 302.3991089
7: -169.8127136, 142.5987701, -165.4024200, 138.8612366, -308.6739502, 308.0011597
8: -204.3698730, 139.3732452, -198.9608154, 135.6417542, -340.0115967, 338.3340454
9: -154.6265411, 152.4968719, -150.5903473, 148.4754181, -303.1019592, 303.0871887

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5984916, upper bound: 330.5993622
time: 7.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5959855, upper bound: 330.5959779
time: 8.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -172.5387726, 137.1988525, -167.6765747, 133.3079681, -305.8467102, 304.8754272
1: -144.7363434, 121.8576889, -140.5860596, 118.3192825, -263.0555725, 262.4436646
2: -190.2408142, 123.3887787, -184.8341675, 119.9155579, -310.1563721, 308.2229004
3: -202.1720581, 106.6576767, -196.3350830, 103.5051041, -305.6771545, 302.9927368
4: -185.9025269, 142.4870605, -180.6953278, 138.4561615, -324.3586731, 323.1823730
5: -166.2956848, 129.2761841, -161.5666199, 125.6717987, -291.9674377, 290.8428040
6: -159.4114532, 153.0630035, -154.9222412, 148.6433411, -308.0547180, 307.9852295
7: -173.1565552, 145.3849335, -168.2875824, 141.2762146, -314.4327698, 313.6724854
8: -208.3753204, 142.0991516, -202.4765320, 138.0367432, -346.4120483, 344.5756836
9: -157.6513977, 155.4814606, -153.2046509, 151.0530701, -308.7044373, 308.6860657

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
time: 7.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
time: 8.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -170.2593536, 135.3842010, -167.3471985, 133.0397949, -303.2991333, 302.7313843
1: -142.8004913, 120.2233505, -140.2787628, 118.0472107, -260.8476868, 260.5021057
2: -187.7037811, 121.7516251, -184.4473724, 119.6526337, -307.3563843, 306.1989746
3: -199.4737854, 105.2304993, -195.9205627, 103.2641907, -302.7379761, 301.1510315
4: -183.4582214, 140.6020050, -180.3611298, 138.1623077, -321.6205444, 320.9631348
5: -164.0971375, 127.5712891, -161.2535095, 125.3985138, -289.4956360, 288.8247986
6: -157.2998810, 151.0235901, -154.6218262, 148.3286896, -305.6285706, 305.6453552
7: -170.8542480, 143.4660492, -167.9489594, 140.9944611, -311.8486938, 311.4149780
8: -205.6142578, 140.2128601, -202.0733795, 137.7600098, -343.3742371, 342.2862549
9: -155.5674133, 153.4230804, -152.9103394, 150.7436523, -306.3110657, 306.3333740

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541
time: 6.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541
time: 10.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.38 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6062120, upper bound: 330.6062469
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6058197, upper bound: 330.6056798
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6062120, upper bound: 330.6062469
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6058197, upper bound: 330.6056798
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6114898, upper bound: 330.6116002
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6111270, upper bound: 330.6111304
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6114898, upper bound: 330.6116002
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6111270, upper bound: 330.6111304
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115183
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115184
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115183
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.5991811, upper bound: 330.5998955
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.5966141, upper bound: 330.5963964
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.5984916, upper bound: 330.5993622
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.5959855, upper bound: 330.5959779
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.38
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -156.6922913, 124.5835342, -171.1702423, 136.1472015, -292.8394775, 295.7537231
1: -131.3162994, 110.5438309, -143.6788940, 120.9657364, -252.2820129, 254.2227173
2: -172.6663666, 112.0884171, -188.8480682, 122.4910049, -295.1573486, 300.9364929
3: -183.3935547, 96.7025604, -200.5681458, 105.7935257, -289.1870728, 297.2706909
4: -168.9003448, 129.3965454, -184.5082703, 141.4546356, -310.3549194, 313.9048157
5: -150.9971466, 117.4427948, -164.9887238, 128.3593750, -279.3565063, 282.4314880
6: -144.7611542, 138.9070435, -158.2664490, 151.8697510, -296.6309204, 297.1734009
7: -157.2247925, 132.0263062, -171.9418335, 144.2940063, -301.5187988, 303.9681396
8: -189.1792145, 128.9833374, -206.7740479, 140.9936066, -330.1727600, 335.7573242
9: -143.1852722, 141.1905975, -156.4783173, 154.3212891, -297.5065613, 297.6689148

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6068285, upper bound: 330.6065470
time: 10.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6050657, upper bound: 330.6053090
time: 7.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -156.1620483, 124.1530762, -169.0881042, 134.4904633, -290.6524658, 293.2411804
1: -130.8342438, 110.1226425, -141.9108582, 119.4739532, -250.3081970, 252.0335083
2: -172.0527649, 111.6792908, -186.5331573, 120.9962540, -293.0490112, 298.2124329
3: -182.7389832, 96.3352890, -198.1059570, 104.4923553, -287.2313232, 294.4411926
4: -168.3441467, 128.9275665, -182.2758331, 139.7327881, -308.0768738, 311.2033997
5: -150.4855194, 117.0124283, -162.9826508, 126.8013687, -277.2868958, 279.9949951
6: -144.2655334, 138.4090881, -156.3385315, 150.0096893, -294.2752075, 294.7475891
7: -156.6768951, 131.5685272, -169.8413544, 142.5412445, -299.2180786, 301.4098816
8: -188.5316010, 128.5376892, -204.2512970, 139.2707214, -327.8023071, 332.7890015
9: -142.7012177, 140.6940765, -154.5736847, 152.4420471, -295.1431885, 295.2677612

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6064224, upper bound: 330.6059732
time: 7.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6047413, upper bound: 330.6047412
time: 7.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -156.6922913, 124.5835342, -177.2374268, 140.9686432, -297.6609497, 301.8209534
1: -131.3162994, 110.5438309, -148.7938995, 125.2744904, -256.5907898, 259.3377380
2: -172.6663666, 112.0884171, -195.5375977, 126.7829514, -299.4493103, 307.6260071
3: -183.3935547, 96.7025604, -207.7729950, 109.6246948, -293.0182495, 304.4755554
4: -168.9003448, 129.3965454, -190.9983215, 146.4466858, -315.3470154, 320.3948364
5: -150.9971466, 117.4427948, -170.8375244, 132.8725433, -283.8695984, 288.2803345
6: -144.7611542, 138.9070435, -163.8513031, 157.2749634, -302.0361328, 302.7583618
7: -157.2247925, 132.0263062, -177.9784088, 149.4057770, -306.6305542, 310.0046997
8: -189.1792145, 128.9833374, -214.1383362, 146.0240784, -335.2033081, 343.1216431
9: -143.1852722, 141.1905975, -161.9952850, 159.7733765, -302.9586487, 303.1858215

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6060654, upper bound: 330.6063671
time: 8.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6060654, upper bound: 330.6116000
time: 9.19 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -156.1620483, 124.1530762, -175.1049194, 139.2720947, -295.4340820, 299.2579956
1: -130.8342438, 110.1226425, -146.9832916, 123.7459106, -254.5801544, 257.1059265
2: -172.0527649, 111.6792908, -193.1662903, 125.2519836, -297.3047485, 304.8455505
3: -182.7389832, 96.3352890, -205.2488403, 108.2884674, -291.0274658, 301.5841064
4: -168.3441467, 128.9275665, -188.7124023, 144.6831360, -313.0272522, 317.6399231
5: -150.4855194, 117.0124283, -168.7821503, 131.2772980, -281.7628174, 285.7945251
6: -144.2655334, 138.4090881, -161.8768005, 155.3688049, -299.6343384, 300.2858887
7: -156.6768951, 131.5685272, -175.8283081, 147.6107788, -304.2876587, 307.3967896
8: -188.5316010, 128.5376892, -211.5529480, 144.2590790, -332.7906799, 340.0906372
9: -142.7012177, 140.6940765, -160.0455627, 157.8479614, -300.5491638, 300.7396240

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6057522, upper bound: 330.6058968
time: 7.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6057522, upper bound: 330.6111260
time: 8.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -156.8526611, 124.7644653, -173.3167419, 137.8356323, -294.6882629, 298.0811768
1: -131.5747223, 110.8021164, -145.4371490, 122.4361725, -254.0108948, 256.2392578
2: -172.9314728, 112.3184738, -191.1693268, 123.9951248, -296.9265747, 303.4877930
3: -183.6140442, 96.9212494, -203.0345764, 107.0943069, -290.7082825, 299.9558105
4: -169.0901489, 129.6018372, -186.7693634, 143.1620636, -312.2521973, 316.3711548
5: -151.2193298, 117.5645828, -167.0489807, 129.9056549, -281.1249695, 284.6135559
6: -144.9535065, 139.1769104, -160.1628265, 153.7571259, -298.7106323, 299.3397217
7: -157.4647827, 132.1934662, -174.0298615, 146.0383759, -303.5031433, 306.2233276
8: -189.3725433, 129.1676636, -209.3224182, 142.7457581, -332.1182861, 338.4899902
9: -143.3786774, 141.4527283, -158.3818665, 156.1977539, -299.5764160, 299.8345947

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6067434, upper bound: 330.6064190
time: 8.24 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6063088, upper bound: 330.6061393
time: 8.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -160.0070801, 127.2517471, -173.3167419, 137.8356323, -297.8426514, 300.5684204
1: -134.1806183, 112.9787064, -145.4371490, 122.4361725, -256.6167908, 258.4158630
2: -176.3727112, 114.5259705, -191.1693268, 123.9951248, -300.3677979, 305.6953125
3: -187.2828979, 98.8370590, -203.0345764, 107.0943069, -294.3771667, 301.8716431
4: -172.4374695, 132.1474609, -186.7693634, 143.1620636, -315.5995178, 318.9168091
5: -154.2406616, 119.8856354, -167.0489807, 129.9056549, -284.1463013, 286.9345703
6: -147.8032684, 141.9399109, -160.1628265, 153.7571259, -301.5603943, 302.1027222
7: -160.5737305, 134.8005829, -174.0298615, 146.0383759, -306.6121216, 308.8304443
8: -193.1587830, 131.7485504, -209.3224182, 142.7457581, -335.9045410, 341.0709534
9: -146.2083740, 144.2302856, -158.3818665, 156.1977539, -302.4061279, 302.6121216

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6067434, upper bound: 330.6064190
time: 7.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6063088, upper bound: 330.6061393
time: 8.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -156.8526611, 124.7644653, -179.5378876, 142.7784882, -299.6311340, 304.3023682
1: -131.5747223, 110.8021164, -150.6759186, 126.8520966, -258.4268188, 261.4780273
2: -172.9314728, 112.3184738, -198.0240936, 128.3885803, -301.3200684, 310.3424988
3: -183.6140442, 96.9212494, -210.4280396, 111.0190048, -294.6330261, 307.3493042
4: -169.0901489, 129.6018372, -193.4234619, 148.2806091, -317.3707581, 323.0252991
5: -151.2193298, 117.5645828, -173.0454254, 134.5266724, -285.7459412, 290.6100159
6: -144.9535065, 139.1769104, -165.8876343, 159.2951660, -304.2486572, 305.0645447
7: -157.4647827, 132.1934662, -180.2161713, 151.2738495, -308.7386475, 312.4096375
8: -189.3725433, 129.1676636, -216.8638916, 147.9069824, -337.2794495, 346.0315247
9: -143.3786774, 141.4527283, -164.0352478, 161.7828217, -305.1614685, 305.4879761

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6062885, upper bound: 330.6060164
time: 8.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6058197, upper bound: 330.6056798
time: 8.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.26 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6068285, upper bound: 330.6065470
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6050657, upper bound: 330.6053090
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6064224, upper bound: 330.6059732
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6047413, upper bound: 330.6047412
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6060654, upper bound: 330.6063671
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6060654, upper bound: 330.6116000
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6057522, upper bound: 330.6058968
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6057522, upper bound: 330.6111260
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6067434, upper bound: 330.6064190
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6063088, upper bound: 330.6061393
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6067434, upper bound: 330.6064190
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6063088, upper bound: 330.6061393
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6062885, upper bound: 330.6060164
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 18.26
Output dim: 9, lower bound: -330.6058197, upper bound: 330.6056798
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6117920, upper bound: 330.6115183
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6164077, upper bound: 330.6164245
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6114105, upper bound: 330.6113192
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.26
Output dim: 9, lower bound: -330.6109541, upper bound: 330.6109541
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=333.2668151855469
rel_dist={9: [-330.62209603555596, 330.6220960272268]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1804.81 seconds
