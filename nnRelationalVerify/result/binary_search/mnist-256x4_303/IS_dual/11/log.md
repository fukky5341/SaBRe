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
execution time: IAR + LP analysis = 1.13 + 10.42 = 11.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -330.6222845, upper bound: 330.6222845


# Binary Search by BASE starts (time budget: 2688.45 seconds, max iter: 100)

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
Binary search time: 48.21 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2640.24 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6148740, upper bound: 330.6145330
time: 6.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221776
time: 8.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.98 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.98
Output dim: 9, lower bound: -330.6148740, upper bound: 330.6145330
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.98
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221776

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -197.7375183, 157.2953796, -183.0468445, 145.5493622, -343.2868042, 340.3421936
1: -166.1268463, 139.7890320, -153.6255188, 129.3070831, -295.4339294, 293.4145508
2: -218.2187042, 141.4309998, -201.9341736, 130.9277802, -349.1464844, 343.3651428
3: -231.8148499, 122.2137833, -214.5020752, 113.1059036, -344.9207458, 336.7158508
4: -213.0580750, 163.3319397, -197.2194214, 151.1820526, -364.2401123, 360.5513611
5: -190.5903168, 148.2323761, -176.3987122, 137.1923523, -327.7826538, 324.6311035
6: -182.8186951, 175.4655914, -169.1759033, 162.3715363, -345.1901855, 344.6414795
7: -198.6573944, 166.6247406, -183.8370056, 154.2434235, -352.9008179, 350.4617310
8: -238.9674683, 162.8750153, -221.0872498, 150.7401733, -389.7076416, 383.9622803
9: -180.6529846, 178.2241058, -167.2702026, 164.9339752, -345.5869446, 345.4942932

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6044969, upper bound: 330.6043183
time: 6.40 seconds

## Relational analysis of IS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6145834, upper bound: 330.6142610
time: 7.19 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6148714, upper bound: 330.6145330
time: 7.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -183.2046204, 145.6741486, -183.6371460, 146.0147095, -329.2192993, 329.3112488
1: -153.7576599, 129.4180298, -154.1186218, 129.7214966, -283.4791260, 283.5366516
2: -202.1076813, 131.0395203, -202.5823517, 131.3443756, -333.4520264, 333.6218262
3: -214.6872711, 113.2025070, -215.1947327, 113.4665604, -328.1537781, 328.3972473
4: -197.3893280, 151.3109283, -197.8540955, 151.6625977, -349.0519409, 349.1650391
5: -176.5500793, 137.3100433, -176.9648895, 137.6301270, -314.1802063, 314.2749023
6: -169.3202972, 162.5115356, -169.7169952, 162.8930969, -332.2133179, 332.2285156
7: -183.9946136, 154.3760223, -184.4264832, 154.7376404, -338.7322388, 338.8024597
8: -221.2767487, 150.8683472, -221.7955170, 151.2197571, -372.4965210, 372.6638794
9: -167.4136353, 165.0750885, -167.8070984, 165.4597626, -332.8734131, 332.8820801

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221477
time: 7.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221776
time: 7.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.45 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 16.45
Output dim: 9, lower bound: -330.6145834, upper bound: 330.6142610
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 16.45
Output dim: 9, lower bound: -330.6148714, upper bound: 330.6145330
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.45
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221477
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.45
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221776

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -193.6001434, 154.0341949, -182.5261536, 145.1374664, -338.7376099, 336.5603638
1: -162.7017059, 136.9298706, -153.1907043, 128.9434814, -291.6452026, 290.1205444
2: -213.6992950, 138.5310669, -201.3608246, 130.5579681, -344.2572632, 339.8918762
3: -226.9907837, 119.7000656, -213.8902283, 112.7867508, -339.7775269, 333.5901794
4: -208.6488495, 159.9816284, -196.6591644, 150.7567596, -359.4055786, 356.6407776
5: -186.6270752, 145.1862793, -175.8973541, 136.8050385, -323.4320679, 321.0836182
6: -179.0664215, 171.8344879, -168.6980743, 161.9110107, -340.9773560, 340.5325623
7: -194.5652924, 163.2007599, -183.3133850, 153.8073425, -348.3726196, 346.5140991
8: -233.9955597, 159.4903107, -220.4617310, 150.3179016, -384.3134155, 379.9520264
9: -176.9279938, 174.5740814, -166.7971649, 164.4684296, -341.3963928, 341.3711548

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6032026, upper bound: 330.6032960
time: 6.94 seconds

## Relational analysis of IS_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6141582, upper bound: 330.6139806
time: 8.82 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6145834, upper bound: 330.6142610
time: 7.99 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -196.8033905, 156.5570831, -183.0468445, 145.5493622, -342.3527527, 339.6039429
1: -165.3444824, 139.1353760, -153.6255188, 129.3070831, -294.6515503, 292.7608643
2: -217.1899872, 140.7682800, -201.9341736, 130.9277802, -348.1177673, 342.7023926
3: -230.7168732, 121.6426392, -214.5020752, 113.1059036, -343.8227844, 336.1446228
4: -212.0521240, 162.5680847, -197.2194214, 151.1820526, -363.2341919, 359.7875061
5: -189.6907959, 147.5392456, -176.3987122, 137.1923523, -326.8830261, 323.9379578
6: -181.9603882, 174.6375427, -169.1759033, 162.3715363, -344.3319092, 343.8134155
7: -197.7184601, 165.8423157, -183.8370056, 154.2434235, -351.9618835, 349.6793213
8: -237.8455505, 162.1158447, -221.0872498, 150.7401733, -388.5856628, 383.2030945
9: -179.8036804, 177.3873596, -167.2702026, 164.9339752, -344.7376099, 344.6575623

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6044969, upper bound: 330.6043183
time: 8.12 seconds

## Relational analysis of IS_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6111241, upper bound: 330.6108179
time: 7.54 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6035008, upper bound: 330.6019322
time: 6.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -182.5547028, 145.1595612, -181.6542053, 144.4335480, -326.9882202, 326.8137817
1: -153.2162170, 128.9624786, -152.4877777, 128.3358307, -281.5520630, 281.4502563
2: -201.3933411, 130.5771484, -200.4157867, 129.9393616, -331.3327026, 330.9928284
3: -213.9243622, 112.8042603, -212.8654938, 112.2496109, -326.1738586, 325.6696472
4: -196.6928101, 150.7800751, -195.7333069, 150.0527954, -346.7455750, 346.5133667
5: -175.9249420, 136.8291931, -175.0541077, 136.1648560, -312.0897827, 311.8832703
6: -168.7238922, 161.9377594, -167.9061432, 161.1460724, -329.8699341, 329.8439026
7: -183.3448639, 153.8305359, -182.4640961, 153.0643311, -336.4091797, 336.2945557
8: -220.4981384, 150.3399200, -219.4412537, 149.6216125, -370.1197510, 369.7811890
9: -166.8218536, 164.4960938, -166.0018768, 163.7009430, -330.5226746, 330.4979553

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5941679, upper bound: 330.5907201
time: 8.15 seconds

## Relational analysis of IS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6053159, upper bound: 330.6035122
time: 9.27 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221477
time: 7.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -183.2046204, 145.6741486, -182.0936584, 144.7929535, -327.9975586, 327.7677307
1: -153.7576599, 129.4180298, -152.8318329, 128.6406860, -282.3983459, 282.2498779
2: -202.1076813, 131.0395203, -200.8863068, 130.2440186, -332.3516235, 331.9258118
3: -214.6872711, 113.2025070, -213.3836365, 112.5203094, -327.2074890, 326.5861511
4: -197.3893280, 151.3109283, -196.2001038, 150.4030151, -347.7922974, 347.5110474
5: -176.5500793, 137.3100433, -175.4801941, 136.4890900, -313.0391541, 312.7902222
6: -169.3202972, 162.5115356, -168.3006592, 161.5304871, -330.8507080, 330.8121948
7: -183.9946136, 154.3760223, -182.8839874, 153.4417419, -337.4363403, 337.2599792
8: -221.2767487, 150.8683472, -219.9491425, 149.9669647, -371.2437134, 370.8174744
9: -167.4136353, 165.0750885, -166.4031067, 164.0850067, -331.4985962, 331.4782104

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5728556, upper bound: 330.5795204
time: 6.94 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5561179, upper bound: 330.5561179
time: 6.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 14.43 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.6141582, upper bound: 330.6139806
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.6145834, upper bound: 330.6142610
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.6111241, upper bound: 330.6108179
IS_A1_A2_A2, status: Status.VERIFIED, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.6035008, upper bound: 330.6019322
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.6053159, upper bound: 330.6035122
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.6221776, upper bound: 330.6221477
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.5728556, upper bound: 330.5795204
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 14.43
Output dim: 9, lower bound: -330.5561179, upper bound: 330.5561179

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -191.9737549, 152.7436218, -181.8773041, 144.6237640, -336.5975037, 334.6208801
1: -161.3790588, 135.8037262, -152.6501770, 128.4887390, -289.8677979, 288.4538879
2: -211.9305115, 137.3812714, -200.6477203, 130.0963745, -342.0268860, 338.0288696
3: -225.0831451, 118.7043152, -213.1286621, 112.3891907, -337.4723511, 331.8329468
4: -206.9184723, 158.6748352, -195.9638062, 150.2268066, -357.1452637, 354.6385803
5: -185.0643616, 143.9975739, -175.2732849, 136.3249969, -321.3893127, 319.2708740
6: -177.5924988, 170.4090118, -168.1026764, 161.3381805, -338.9306641, 338.5116577
7: -192.9702301, 161.8344421, -182.6647644, 153.2628326, -346.2330017, 344.4991760
8: -232.0744171, 158.1855011, -219.6844330, 149.7904053, -381.8647461, 377.8699341
9: -175.4496155, 173.1407776, -166.2063751, 163.8905029, -339.3400269, 339.3470764

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6020499, upper bound: 330.6023403
time: 8.28 seconds

## Relational analysis of IS_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6026339, upper bound: 330.6015839
time: 7.58 seconds

## Relational analysis of IS_A1_A1_A1_A2

### Relational analysis result of IS_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6026339, upper bound: 330.6064007
time: 8.17 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -192.0575714, 152.8137360, -182.5261536, 145.1374664, -337.1949768, 335.3398438
1: -161.4160767, 135.8497467, -153.1907043, 128.9434814, -290.3595581, 289.0404053
2: -212.0044556, 137.4321442, -201.3608246, 130.5579681, -342.5624390, 338.7929688
3: -225.1820984, 118.7541046, -213.8902283, 112.7867508, -337.9687805, 332.6442871
4: -206.9963074, 158.7230835, -196.6591644, 150.7567596, -357.7530518, 355.3822327
5: -185.1436005, 144.0453949, -175.8973541, 136.8050385, -321.9485779, 319.9427490
6: -177.6513214, 170.4726105, -168.6980743, 161.9110107, -339.5622864, 339.1706848
7: -193.0247498, 161.9059448, -183.3133850, 153.8073425, -346.8320618, 345.2192993
8: -232.1507568, 158.2373352, -220.4617310, 150.3179016, -382.4686584, 378.6990662
9: -175.5245972, 173.2004547, -166.7971649, 164.4684296, -339.9930115, 339.9975891

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_A2_A1

### Relational analysis result of IS_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6032026, upper bound: 330.6032960
time: 6.58 seconds

## Relational analysis of IS_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_A2_A1

### Relational analysis result of IS_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6032918, upper bound: 330.6021986
time: 7.65 seconds

## Relational analysis of IS_A1_A1_A2_A2

### Relational analysis result of IS_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6074774, upper bound: 330.6068422
time: 6.70 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -194.3016052, 154.5834503, -183.0468445, 145.5493622, -339.8509521, 337.6303101
1: -163.2557831, 137.3885803, -153.6255188, 129.3070831, -292.5628662, 291.0140991
2: -214.4427643, 138.9936981, -201.9341736, 130.9277802, -345.3705139, 340.9278564
3: -227.7828674, 120.1165619, -214.5020752, 113.1059036, -340.8887634, 334.6185913
4: -209.3715057, 160.5218658, -197.2194214, 151.1820526, -360.5535583, 357.7412720
5: -187.2903900, 145.6835327, -176.3987122, 137.1923523, -324.4826660, 322.0821838
6: -179.6645813, 172.4367371, -169.1759033, 162.3715363, -342.0361023, 341.6126099
7: -195.2174377, 163.7545166, -183.8370056, 154.2434235, -349.4608765, 347.5915222
8: -234.8537903, 160.0783081, -221.0872498, 150.7401733, -385.5939636, 381.1655579
9: -177.5328827, 175.1611786, -167.2702026, 164.9339752, -342.4668579, 342.4313049

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5842387, upper bound: 330.5853281
time: 7.81 seconds

## Relational analysis of IS_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6107465, upper bound: 330.6105125
time: 7.66 seconds

## Relational analysis of IS_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6111241, upper bound: 330.6108179
time: 10.04 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -182.4125061, 145.0475159, -181.6542053, 144.4335480, -326.8460388, 326.7017212
1: -153.0977173, 128.8627167, -152.4877777, 128.3358307, -281.4335327, 281.3504639
2: -201.2373657, 130.4763947, -200.4157867, 129.9393616, -331.1767273, 330.8921509
3: -213.7572937, 112.7170258, -212.8654938, 112.2496109, -326.0068054, 325.5824585
4: -196.5394592, 150.6639099, -195.7333069, 150.0527954, -346.5922546, 346.3972168
5: -175.7884979, 136.7239990, -175.0541077, 136.1648560, -311.9533691, 311.7780457
6: -168.5928040, 161.8122864, -167.9061432, 161.1460724, -329.7388611, 329.7184448
7: -183.2023010, 153.7116089, -182.4640961, 153.0643311, -336.2665405, 336.1756592
8: -220.3265381, 150.2232361, -219.4412537, 149.6216125, -369.9481506, 369.6644897
9: -166.6920929, 164.3691559, -166.0018768, 163.7009430, -330.3930359, 330.3710022

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5941679, upper bound: 330.5907201
time: 8.21 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5741814, upper bound: 330.5681558
time: 7.96 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221534, upper bound: 330.6221205
time: 10.75 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221484, upper bound: 330.6221036
time: 9.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 45.45 seconds
IS_A1_A1_A1_A1, status: Status.VERIFIED, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6026339, upper bound: 330.6015839
IS_A1_A1_A1_A2, status: Status.VERIFIED, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6026339, upper bound: 330.6064007
IS_A1_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6032918, upper bound: 330.6021986
IS_A1_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6074774, upper bound: 330.6068422
IS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6107465, upper bound: 330.6105125
IS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6111241, upper bound: 330.6108179
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6221534, upper bound: 330.6221205
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 45.45
Output dim: 9, lower bound: -330.6221484, upper bound: 330.6221036

## BFS IS instance: IS_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -192.7313232, 153.3375397, -182.3978424, 145.0355225, -337.7667847, 335.7353821
1: -161.9806824, 136.3004608, -153.0848389, 128.8522034, -290.8328857, 289.3852844
2: -212.7369537, 137.8827667, -201.2208405, 130.4660797, -343.2030029, 339.1035461
3: -225.9367218, 119.1565247, -213.7402496, 112.7082291, -338.6449280, 332.8967896
4: -207.7033081, 159.2605896, -196.5238953, 150.6519318, -358.3552246, 355.7844849
5: -185.7828217, 144.5367126, -175.7744598, 136.7121735, -322.4949951, 320.3111572
6: -178.2411346, 171.0611725, -168.5803528, 161.7985840, -340.0397339, 339.6415405
7: -193.6760406, 162.4357452, -183.1882019, 153.6987457, -347.3746948, 345.6239014
8: -232.9969330, 158.8201447, -220.3097687, 150.2125092, -383.2094421, 379.1299133
9: -176.1047516, 173.7773132, -166.6792908, 164.3558502, -340.4606018, 340.4565735

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5764129, upper bound: 330.5777379
time: 9.18 seconds

## Relational analysis of IS_A1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5983489, upper bound: 330.5974108
time: 7.68 seconds

## Relational analysis of IS_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6033308, upper bound: 330.6030393
time: 8.50 seconds

## BFS IS instance: IS_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -192.7195587, 153.3307190, -183.0468445, 145.5493622, -338.2688904, 336.3775635
1: -161.9362640, 136.2800903, -153.6255188, 129.3070831, -291.2433472, 289.9056091
2: -212.7039490, 137.8652344, -201.9341736, 130.9277802, -343.6316833, 339.7993164
3: -225.9261932, 119.1459274, -214.5020752, 113.1059036, -339.0320740, 333.6479187
4: -207.6763916, 159.2300262, -197.2194214, 151.1820526, -358.8584595, 356.4494629
5: -185.7680054, 144.5127869, -176.3987122, 137.1923523, -322.9603577, 320.9114990
6: -178.2120972, 171.0397339, -169.1759033, 162.3715363, -340.5835876, 340.2156372
7: -193.6362915, 162.4256439, -183.8370056, 154.2434235, -347.8796997, 346.2626343
8: -232.9598083, 158.7923279, -221.0872498, 150.7401733, -383.6999817, 379.8795776
9: -176.0932465, 173.7513580, -167.2702026, 164.9339752, -341.0272217, 341.0215454

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 140

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5842387, upper bound: 330.5853281
time: 8.15 seconds

## Relational analysis of IS_A1_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5989776, upper bound: 330.5980084
time: 8.87 seconds

## Relational analysis of IS_A1_A2_A1_A2_A2

### Relational analysis result of IS_A1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6039820, upper bound: 330.6035464
time: 8.82 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -178.3844910, 141.8719025, -181.1260376, 144.0158844, -322.4003906, 322.9979248
1: -149.7612000, 126.0770035, -152.0468292, 127.9670868, -277.7282410, 278.1238403
2: -196.8329773, 127.6500549, -199.8342133, 129.5643463, -326.3973389, 327.4842529
3: -209.0628662, 110.2726440, -212.2452240, 111.9259796, -320.9888000, 322.5178223
4: -192.2463837, 147.4067230, -195.1651611, 149.6215057, -341.8678589, 342.5718689
5: -171.9265442, 133.7575684, -174.5457306, 135.7720184, -307.6985474, 308.3032837
6: -164.9461060, 158.2740479, -167.4216003, 160.6789703, -325.6250305, 325.6955566
7: -179.2119751, 150.3763275, -181.9332275, 152.6220703, -331.8339539, 332.3094482
8: -215.4913483, 146.9363098, -218.8068390, 149.1934509, -364.6847534, 365.7431641
9: -163.0692444, 160.8122559, -165.5221100, 163.2289886, -326.2982178, 326.3343506

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5866968, upper bound: 330.5830657
time: 7.34 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6158635, upper bound: 330.6155888
time: 8.86 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127045, upper bound: 330.6125378
time: 9.47 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -181.4602509, 144.2946930, -181.6542053, 144.4335480, -325.8937988, 325.9489136
1: -152.3008575, 128.1967163, -152.4877777, 128.3358307, -280.6366882, 280.6844788
2: -200.1882935, 129.8009949, -200.4157867, 129.9393616, -330.1276550, 330.2167053
3: -212.6381226, 112.1354218, -212.8654938, 112.2496109, -324.8876953, 325.0008850
4: -195.5136566, 149.8856354, -195.7333069, 150.0527954, -345.5664673, 345.6189575
5: -174.8719330, 136.0169220, -175.0541077, 136.1648560, -311.0368042, 311.0709839
6: -167.7183228, 160.9682617, -167.9061432, 161.1460724, -328.8643799, 328.8743896
7: -182.2451477, 152.9140472, -182.4640961, 153.0643311, -335.3094177, 335.3780823
8: -219.1828461, 149.4497375, -219.4412537, 149.6216125, -368.8044434, 368.8909912
9: -165.8264465, 163.5163269, -166.0018768, 163.7009430, -329.5273438, 329.5181885

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5941679, upper bound: 330.5907201
time: 8.60 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5741814, upper bound: 330.5681558
time: 8.69 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6159201, upper bound: 330.6156046
time: 7.96 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6152669, upper bound: 330.6152677
time: 8.67 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6127045, upper bound: 330.6125378
time: 8.27 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 55.98 seconds
IS_A1_A2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.5983489, upper bound: 330.5974108
IS_A1_A2_A1_A1_A2, status: Status.VERIFIED, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.6033308, upper bound: 330.6030393
IS_A1_A2_A1_A2_A1, status: Status.VERIFIED, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.5989776, upper bound: 330.5980084
IS_A1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.6039820, upper bound: 330.6035464
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.6158635, upper bound: 330.6155888
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.6127045, upper bound: 330.6125378
IS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.6152669, upper bound: 330.6152677
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 55.98
Output dim: 9, lower bound: -330.6127045, upper bound: 330.6125378

## BFS IS instance: IS_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -178.3844910, 141.8719025, -178.1108551, 141.6182709, -320.0027466, 319.9827271
1: -149.7612000, 126.0770035, -149.4827423, 125.8301086, -275.5913086, 275.5597534
2: -196.8329773, 127.6500549, -196.4793091, 127.4188461, -324.2518311, 324.1293335
3: -209.0628662, 110.2726440, -208.7032013, 110.0683746, -319.1311646, 318.9758301
4: -192.2463837, 147.4067230, -191.8767853, 147.1167145, -339.3630371, 339.2835083
5: -171.9265442, 133.7575684, -171.6412048, 133.4944153, -305.4209595, 305.3987732
6: -164.9461060, 158.2740479, -164.6287689, 157.9816589, -322.9277039, 322.9028015
7: -179.2119751, 150.3763275, -178.8853302, 150.0755768, -329.2874451, 329.2616577
8: -215.4913483, 146.9363098, -215.1491089, 146.6994171, -362.1907043, 362.0853882
9: -163.0692444, 160.8122559, -162.7574615, 160.4605255, -323.5297852, 323.5697021

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6130718, upper bound: 330.6128121
time: 9.26 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6110110, upper bound: 330.6106798
time: 8.25 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -176.8155518, 140.6242676, -183.7366333, 145.9915161, -322.8070374, 324.3608704
1: -148.4235229, 124.9643173, -153.9692230, 129.6412506, -278.0646973, 278.9335327
2: -195.0877380, 126.5330200, -202.4905853, 131.3288422, -326.4165649, 329.0236206
3: -207.2178040, 109.3076401, -215.1924896, 113.4458771, -320.6636963, 324.5001221
4: -190.5389099, 146.1051025, -197.7572784, 151.5650940, -342.1040039, 343.8623657
5: -170.4142761, 132.5748596, -177.0052338, 137.5587616, -307.9730225, 309.5800781
6: -163.4934998, 156.8698120, -169.6546936, 162.7950897, -326.2885132, 326.5245056
7: -177.6276093, 149.0525818, -184.3784637, 154.7074890, -332.3350830, 333.4309998
8: -213.5826263, 145.6384888, -221.6897888, 151.1075439, -364.6901855, 367.3282471
9: -161.6326447, 159.3713226, -167.7937622, 165.1539764, -326.7866211, 327.1651001

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5978324, upper bound: 330.5964459
time: 8.07 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5914173, upper bound: 330.5913901
time: 5.30 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -178.4190216, 141.8763275, -181.6542053, 144.4335480, -322.8525696, 323.5305176
1: -149.7139282, 126.0411606, -152.4877777, 128.3358307, -278.0497131, 278.5289307
2: -196.8042450, 127.6355591, -200.4157867, 129.9393616, -326.7435913, 328.0513000
3: -209.0638428, 110.2621460, -212.8654938, 112.2496109, -321.3134155, 323.1276245
4: -192.1965790, 147.3597412, -195.7333069, 150.0527954, -342.2493591, 343.0930481
5: -171.9417725, 133.7200928, -175.0541077, 136.1648560, -308.1066284, 308.7741394
6: -164.9018402, 158.2472839, -167.9061432, 161.1460724, -326.0478821, 326.1534119
7: -179.1706390, 150.3457794, -182.4640961, 153.0643311, -332.2349243, 332.8098450
8: -215.4917297, 146.9349213, -219.4412537, 149.6216125, -365.1133423, 366.3761597
9: -163.0378723, 160.7233582, -166.0018768, 163.7009430, -326.7388000, 326.7252197

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 111

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5711470, upper bound: 330.5692413
time: 8.43 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6152486, upper bound: 330.6152172
time: 9.20 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6152486, upper bound: 330.6152172
time: 8.02 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -184.0017700, 146.2133942, -180.0774231, 143.1798553, -327.1816406, 326.2907410
1: -154.1620636, 129.8223572, -151.1437531, 127.2165451, -281.3785095, 280.9661255
2: -202.7675323, 131.5136566, -198.6610718, 128.8170013, -331.5845337, 330.1746521
3: -215.5035400, 113.6144485, -211.0109253, 111.2789993, -326.7825317, 324.6253357
4: -198.0327759, 151.7705536, -194.0168915, 148.7440033, -346.7767029, 345.7874451
5: -177.2594452, 137.7513580, -173.5349274, 134.9761200, -312.2355652, 311.2862244
6: -169.8923798, 163.0241241, -166.4447021, 159.7346039, -329.6269836, 329.4688110
7: -184.6219177, 154.9395294, -180.8710480, 151.7337646, -336.3556824, 335.8105774
8: -221.9808350, 151.3065033, -217.5221252, 148.3153381, -370.2961731, 368.8286133
9: -168.0332794, 165.3801270, -164.5579834, 162.2528687, -330.2861328, 329.9381104

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6093302, upper bound: 330.6094998
time: 8.21 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6077629, upper bound: 330.6077248
time: 10.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 28.11 seconds
IS_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.6130718, upper bound: 330.6128121
IS_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.6110110, upper bound: 330.6106798
IS_A2_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.5978324, upper bound: 330.5964459
IS_A2_B1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.5914173, upper bound: 330.5913901
IS_A2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.6152486, upper bound: 330.6152172
IS_A2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.6152486, upper bound: 330.6152172
IS_A2_B1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.6093302, upper bound: 330.6094998
IS_A2_B1_A2_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 28.11
Output dim: 9, lower bound: -330.6077629, upper bound: 330.6077248

## BFS IS instance: IS_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.9578857, 139.9535370, -178.1108551, 141.6182709, -317.5761108, 318.0643616
1: -147.7303467, 124.3769150, -149.4827423, 125.8301086, -273.5604248, 273.8596497
2: -194.1604919, 125.9313965, -196.4793091, 127.4188461, -321.5793152, 322.4107056
3: -206.2139282, 108.7842560, -208.7032013, 110.0683746, -316.2822876, 317.4874268
4: -189.6378632, 145.4184418, -191.8767853, 147.1167145, -336.7544861, 337.2952271
5: -169.5897675, 131.9571075, -171.6412048, 133.4944153, -303.0841675, 303.5983276
6: -162.7197876, 156.1271057, -164.6287689, 157.9816589, -320.7013855, 320.7558594
7: -176.7816315, 148.3359070, -178.8853302, 150.0755768, -326.8572083, 327.2212219
8: -212.5680542, 144.9615479, -215.1491089, 146.6994171, -359.2674561, 360.1106567
9: -160.8591766, 158.6392517, -162.7574615, 160.4605255, -321.3197021, 321.3967285

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5974242, upper bound: 330.5953738
time: 8.89 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5919371, upper bound: 330.5908275
time: 8.11 seconds

## BFS IS instance: IS_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -189.5727997, 150.6585999, -177.6578217, 141.2603912, -330.8331909, 328.3164062
1: -159.1587982, 133.9192810, -149.1038208, 125.5130615, -284.6718750, 283.0231018
2: -209.1500702, 135.5968933, -195.9804688, 127.0986786, -336.2486877, 331.5773010
3: -222.1014099, 117.0402756, -208.1715851, 109.7904205, -331.8918457, 325.2118530
4: -204.1927643, 156.5574799, -191.3897247, 146.7456818, -350.9384155, 347.9471436
5: -182.6185913, 142.0281677, -171.2048645, 133.1585388, -315.7771301, 313.2330322
6: -175.2435455, 168.1061554, -164.2126312, 157.5812225, -332.8247681, 332.3187561
7: -190.4128265, 159.7137146, -178.4318085, 149.6950378, -340.1078491, 338.1454773
8: -228.9267883, 156.0123291, -214.6040344, 146.3306580, -375.2574463, 370.6162720
9: -173.1732178, 170.6980133, -162.3444214, 160.0556030, -333.2287903, 333.0424194

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5975999, upper bound: 330.5972660
time: 7.14 seconds

## Relational analysis of IS_A2_B1_A2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5816777, upper bound: 330.5787787
time: 7.59 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 15.91 seconds
IS_A2_B1_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 15.91
Output dim: 9, lower bound: -330.5974242, upper bound: 330.5953738
IS_A2_B1_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 15.91
Output dim: 9, lower bound: -330.5919371, upper bound: 330.5908275
IS_A2_B1_A2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 15.91
Output dim: 9, lower bound: -330.5975999, upper bound: 330.5972660
IS_A2_B1_A2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 15.91
Output dim: 9, lower bound: -330.5816777, upper bound: 330.5787787
IS_A2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.91
Output dim: 9, lower bound: -330.6152486, upper bound: 330.6152172
IS_A2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.91
Output dim: 9, lower bound: -330.6152486, upper bound: 330.6152172
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=333.2668151855469
rel_dist={9: [-330.6221776084849, 330.6221776084849]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5654766, upper bound: 330.5696068
time: 8.86 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5559053, upper bound: 330.5559053
time: 5.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.52 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 14.52
Output dim: 9, lower bound: -330.5654766, upper bound: 330.5696068
IS_A2, status: Status.VERIFIED, split count: 1, time: 14.52
Output dim: 9, lower bound: -330.5559053, upper bound: 330.5559053
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=333.2668151855469
rel_dist={9: [-330.62201561424183, 330.62201560609844]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5682221, upper bound: 330.5733489
time: 8.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5559863, upper bound: 330.5559863
time: 6.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.80 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 14.80
Output dim: 9, lower bound: -330.5682221, upper bound: 330.5733489
IS_A2, status: Status.VERIFIED, split count: 1, time: 14.80
Output dim: 9, lower bound: -330.5559863, upper bound: 330.5559863
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=333.2668151855469
rel_dist={9: [-330.62209603555596, 330.6220960272268]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6141263, upper bound: 330.6144606
time: 6.89 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6221542, upper bound: 330.6221542
time: 7.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.38 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 14.38
Output dim: 9, lower bound: -330.6141263, upper bound: 330.6144606
IS_B2, status: Status.UNKNOWN, split count: 1, time: 14.38
Output dim: 9, lower bound: -330.6221542, upper bound: 330.6221542

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -182.8501434, 145.3943176, -197.7375183, 157.2953796, -340.1454773, 343.1317139
1: -153.4612274, 129.1690063, -166.1268463, 139.7890320, -293.2502441, 295.2958374
2: -201.7181244, 130.7889404, -218.2187042, 141.4309998, -343.1491089, 349.0076294
3: -214.2712860, 112.9857178, -231.8148499, 122.2137833, -336.4850769, 344.8005066
4: -197.0079193, 151.0218964, -213.0580750, 163.3319397, -360.3398438, 364.0799561
5: -176.2100372, 137.0464630, -190.5903168, 148.2323761, -324.4424133, 327.6367798
6: -168.9956055, 162.1977386, -182.8186951, 175.4655914, -344.4611816, 345.0163574
7: -183.6405487, 154.0787354, -198.6573944, 166.6247406, -350.2652893, 352.7361145
8: -220.8512573, 150.5803223, -238.9674683, 162.8750153, -383.7262268, 389.5477905
9: -167.0913239, 164.7587585, -180.6529846, 178.2241058, -345.3153992, 345.4117432

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6027291, upper bound: 330.6028219
time: 9.23 seconds

## Relational analysis of IS_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6141513
time: 8.23 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6144606
time: 7.60 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -183.6371460, 146.0147095, -183.2046204, 145.6741486, -329.3112488, 329.2192993
1: -154.1186218, 129.7214966, -153.7576599, 129.4180298, -283.5366516, 283.4791260
2: -202.5823517, 131.3443756, -202.1076813, 131.0395203, -333.6218262, 333.4520264
3: -215.1947327, 113.4665604, -214.6872711, 113.2025070, -328.3972473, 328.1538086
4: -197.8540955, 151.6625977, -197.3893280, 151.3109283, -349.1650391, 349.0519409
5: -176.9648895, 137.6301270, -176.5500793, 137.3100433, -314.2749023, 314.1802063
6: -169.7169952, 162.8930969, -169.3202972, 162.5115356, -332.2285156, 332.2133179
7: -184.4264832, 154.7376404, -183.9946136, 154.3760223, -338.8024597, 338.7322388
8: -221.7955170, 151.2197571, -221.2767487, 150.8683472, -372.6638794, 372.4965210
9: -167.8070984, 165.4597626, -167.4136353, 165.0750885, -332.8820801, 332.8734131

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5706763, upper bound: 330.5765354
time: 7.34 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5560552, upper bound: 330.5560552
time: 6.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.83 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 14.83
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6141513
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 14.83
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6144606
IS_B2_A1, status: Status.VERIFIED, split count: 2, time: 14.83
Output dim: 9, lower bound: -330.5706763, upper bound: 330.5765354
IS_B2_A2, status: Status.VERIFIED, split count: 2, time: 14.83
Output dim: 9, lower bound: -330.5560552, upper bound: 330.5560552

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -182.1001282, 144.8010254, -193.6001434, 154.0341949, -336.1343384, 338.4011841
1: -152.8349152, 128.6453094, -162.7017059, 136.9298706, -289.7647400, 291.3470154
2: -200.8923798, 130.2562714, -213.6992950, 138.5310669, -339.4234009, 343.9555664
3: -213.3900146, 112.5260086, -226.9907837, 119.7000656, -333.0899963, 339.5167847
4: -196.2009583, 150.4093628, -208.6488495, 159.9816284, -356.1825867, 359.0581665
5: -175.4879150, 136.4886169, -186.6270752, 145.1862793, -320.6741943, 323.1156311
6: -168.3073730, 161.5344391, -179.0664215, 171.8344879, -340.1418152, 340.6007690
7: -182.8864136, 153.4506683, -194.5652924, 163.2007599, -346.0871582, 348.0159607
8: -219.9502563, 149.9721527, -233.9955597, 159.4903107, -379.4405212, 383.9677124
9: -166.4099426, 164.0882568, -176.9279938, 174.5740814, -340.9840088, 341.0162048

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6009078, upper bound: 330.6006952
time: 9.56 seconds

## Relational analysis of IS_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6091796, upper bound: 330.6094318
time: 8.65 seconds

## Relational analysis of IS_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6092704, upper bound: 330.6102817
time: 7.81 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6141513
time: 8.73 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -182.8501434, 145.3943176, -196.8033905, 156.5570831, -339.4072266, 342.1976624
1: -153.4612274, 129.1690063, -165.3444824, 139.1353760, -292.5965576, 294.5134888
2: -201.7181244, 130.7889404, -217.1899872, 140.7682800, -342.4863892, 347.9789429
3: -214.2712860, 112.9857178, -230.7168732, 121.6426392, -335.9138489, 343.7025757
4: -197.0079193, 151.0218964, -212.0521240, 162.5680847, -359.5759888, 363.0740356
5: -176.2100372, 137.0464630, -189.6907959, 147.5392456, -323.7492676, 326.7372131
6: -168.9956055, 162.1977386, -181.9603882, 174.6375427, -343.6331177, 344.1580811
7: -183.6405487, 154.0787354, -197.7184601, 165.8423157, -349.4828491, 351.7971802
8: -220.8512573, 150.5803223, -237.8455505, 162.1158447, -382.9670410, 388.4257812
9: -167.0913239, 164.7587585, -179.8036804, 177.3873596, -344.4786987, 344.5624084

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6027291, upper bound: 330.6028219
time: 7.57 seconds

## Relational analysis of IS_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6098542, upper bound: 330.6100965
time: 7.66 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6010797, upper bound: 330.6025303
time: 8.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.51 seconds
IS_B1_B1_B1, status: Status.VERIFIED, split count: 3, time: 28.51
Output dim: 9, lower bound: -330.6092704, upper bound: 330.6102817
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 28.51
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6141513
IS_B1_B2_B1, status: Status.VERIFIED, split count: 3, time: 28.51
Output dim: 9, lower bound: -330.6098542, upper bound: 330.6100965
IS_B1_B2_B2, status: Status.VERIFIED, split count: 3, time: 28.51
Output dim: 9, lower bound: -330.6010797, upper bound: 330.6025303

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: -182.1001282, 144.8010254, -192.4550171, 153.1248322, -335.2249756, 337.2560425
1: -152.8349152, 128.6453094, -161.7426453, 136.1250916, -288.9599915, 290.3878784
2: -200.8923798, 130.2562714, -212.4413147, 137.7183228, -338.6106873, 342.6975708
3: -213.3900146, 112.5260086, -225.6434479, 118.9956665, -332.3856812, 338.1694641
4: -196.2009583, 150.4093628, -207.4071045, 159.0396729, -355.2406311, 357.8164368
5: -175.4879150, 136.4886169, -185.5244293, 144.3315430, -319.8194580, 322.0130005
6: -168.3073730, 161.5344391, -178.0191345, 170.8220215, -339.1293640, 339.5534973
7: -182.8864136, 153.4506683, -193.4163818, 162.2381744, -345.1245728, 346.8670654
8: -219.9502563, 149.9721527, -232.6128235, 158.5470734, -378.4972839, 382.5849609
9: -166.4099426, 164.0882568, -175.8811951, 173.5529633, -339.9628906, 339.9694519

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B1_B2_B1

### Relational analysis result of IS_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6009078, upper bound: 330.6006952
time: 8.55 seconds

## Relational analysis of IS_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_B1_B1_B2_B1

### Relational analysis result of IS_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6135434, upper bound: 330.6137109
time: 7.33 seconds

## Relational analysis of IS_B1_B1_B2_B2

### Relational analysis result of IS_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6141513
time: 8.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.41 seconds
IS_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 9, lower bound: -330.6135434, upper bound: 330.6137109
IS_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.41
Output dim: 9, lower bound: -330.6138299, upper bound: 330.6141513

## BFS IS instance: IS_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -181.1540222, 144.0519867, -190.8196411, 151.8266296, -332.9806213, 334.8716431
1: -152.0467377, 127.9821701, -160.4118652, 134.9922485, -287.0390015, 288.3940430
2: -199.8525085, 129.5832520, -210.6623535, 136.5617828, -336.4143066, 340.2455750
3: -212.2794952, 111.9463043, -223.7250519, 117.9940338, -330.2734985, 335.6713562
4: -195.1870575, 149.6365814, -205.6669312, 157.7251282, -352.9121704, 355.3035278
5: -174.5779572, 135.7886200, -183.9523468, 143.1353760, -317.7132874, 319.7409363
6: -167.4391937, 160.6991882, -176.5363770, 169.3885498, -336.8277283, 337.2355347
7: -181.9406128, 152.6566315, -191.8120575, 160.8634186, -342.8040161, 344.4686890
8: -218.8167877, 149.2029572, -230.6809998, 157.2344666, -376.0512390, 379.8839722
9: -165.5485229, 163.2454987, -174.3939819, 172.1113281, -337.6597900, 337.6394348

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B1_B2_B1_B1

### Relational analysis result of IS_B1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.5994460, upper bound: 330.5989869
time: 8.43 seconds

## Relational analysis of IS_B1_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_B1_B2_B1_B1

### Relational analysis result of IS_B1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6007652, upper bound: 330.6017158
time: 8.78 seconds

## Relational analysis of IS_B1_B1_B2_B1_B2

### Relational analysis result of IS_B1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6058445, upper bound: 330.6063856
time: 9.04 seconds

## BFS IS instance: IS_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -182.1001282, 144.8010254, -190.9049072, 151.8985138, -333.9986572, 335.7059326
1: -152.8349152, 128.6453094, -160.4506836, 135.0397034, -287.8746338, 289.0960083
2: -200.8923798, 130.2562714, -210.7382812, 136.6140594, -337.5064087, 340.9945679
3: -213.3900146, 112.5260086, -223.8260040, 118.0450897, -331.4350586, 336.3520203
4: -196.2009583, 150.4093628, -205.7465820, 157.7750702, -353.9760132, 356.1559448
5: -175.4879150, 136.4886169, -184.0337067, 143.1849365, -318.6727600, 320.5222778
6: -168.3073730, 161.5344391, -176.5971375, 169.4535675, -337.7609253, 338.1315308
7: -182.8864136, 153.4506683, -191.8683319, 160.9369965, -343.8233948, 345.3190002
8: -219.9502563, 149.9721527, -230.7590485, 157.2879333, -377.2381897, 380.7312012
9: -166.4099426, 164.0882568, -174.4709320, 172.1727142, -338.5826416, 338.5591431

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_B1_B2_B2_B1

### Relational analysis result of IS_B1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6009078, upper bound: 330.6006952
time: 7.76 seconds

## Relational analysis of IS_B1_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_B1_B2_B2_B1

### Relational analysis result of IS_B1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6013949, upper bound: 330.6023983
time: 8.53 seconds

## Relational analysis of IS_B1_B1_B2_B2_B2

### Relational analysis result of IS_B1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -330.6063166, upper bound: 330.6069603
time: 10.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 29.56 seconds
IS_B1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 29.56
Output dim: 9, lower bound: -330.6007652, upper bound: 330.6017158
IS_B1_B1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 29.56
Output dim: 9, lower bound: -330.6058445, upper bound: 330.6063856
IS_B1_B1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 29.56
Output dim: 9, lower bound: -330.6013949, upper bound: 330.6023983
IS_B1_B1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 29.56
Output dim: 9, lower bound: -330.6063166, upper bound: 330.6069603
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=333.2668151855469
rel_dist={9: [-330.62215419250697, 330.62215419250697]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 895.98 seconds
