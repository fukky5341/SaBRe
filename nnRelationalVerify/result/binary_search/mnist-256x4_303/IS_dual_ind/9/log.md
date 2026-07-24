## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 461.219499622
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213)
1: (-210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735)
2: (-277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723)
3: (-294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076)
4: (-270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802)
5: (-242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328)
6: (-231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295)
7: (-252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404)
8: (-304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772)
9: (-229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984)

## BASE Result
execution time: IAR + LP analysis = 1.13 + 11.62 = 12.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -461.2313825, upper bound: 461.2313824


# Binary Search by BASE starts (time budget: 2687.25 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=464.3514404296875
rel_dist={7: [-461.2313236619183, 461.23132366191817]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=464.3514404296875
rel_dist={7: [-461.23120259765585, 461.23120260223186]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=464.3514404296875
rel_dist={7: [-461.2309993444553, 461.2309993438606]}

## Binary Search Result
Binary search time: 50.37 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2636.88 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2276815, upper bound: 461.2274451
time: 10.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
time: 8.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.69
Output dim: 7, lower bound: -461.2276815, upper bound: 461.2274451
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.69
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -252.2428284, 199.5615692, -438.6359253, 441.4051819
1: -199.7922211, 167.9400635, -210.8920135, 177.2409058, -377.0331116, 378.8320618
2: -262.6058350, 170.4243469, -277.1440125, 179.7689209, -442.3746948, 447.5683289
3: -279.3811340, 147.4806366, -294.8527222, 155.5875702, -434.9686890, 442.3333740
4: -256.2637939, 196.3474579, -270.4500427, 207.2221527, -463.4859619, 466.7974854
5: -229.6463013, 178.5066833, -242.3452606, 188.3783875, -418.0246887, 420.8518372
6: -219.7590637, 210.8946838, -231.8721466, 222.5393829, -442.2984314, 442.7667847
7: -239.1515045, 200.9265594, -252.3833771, 211.9680939, -451.1195679, 453.3099365
8: -288.1658630, 196.6365509, -304.0357361, 207.4447021, -495.6105042, 500.6723022
9: -217.7042542, 214.2737274, -229.7125092, 226.0945892, -443.7988281, 443.9862061

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
time: 7.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
time: 8.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -242.0842743, 191.5462341, -249.6317444, 197.5000153, -439.5842896, 441.1779480
1: -202.2670288, 170.0345459, -208.6938171, 175.4012756, -377.6683044, 378.7283325
2: -265.9084778, 172.5523376, -274.2660217, 177.9192963, -443.8277588, 446.8182983
3: -282.9065552, 149.2875214, -291.7886047, 153.9832001, -436.8897705, 441.0761108
4: -259.4692688, 198.7713928, -267.6379700, 205.0684509, -464.5377197, 466.4093628
5: -232.5094452, 180.7077179, -239.8283081, 186.4242249, -418.9336243, 420.5360107
6: -222.5233765, 213.5542908, -229.4713135, 220.2323761, -442.7557373, 443.0256042
7: -242.1613312, 203.4472351, -249.7637024, 209.7811737, -451.9424744, 453.2108765
8: -291.7500916, 199.0496979, -300.8900452, 205.3010254, -497.0510864, 499.9397278
9: -220.4269104, 216.9405365, -227.3344879, 223.7521362, -444.1790466, 444.2750244

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
time: 9.09 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
time: 8.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -239.0743866, 189.1624146, -428.2367859, 428.2367859
1: -199.7922211, 167.9400635, -199.7922211, 167.9400635, -367.7322998, 367.7322998
2: -262.6058350, 170.4243469, -262.6058350, 170.4243469, -433.0301514, 433.0301514
3: -279.3811340, 147.4806366, -279.3811340, 147.4806366, -426.8617554, 426.8617554
4: -256.2637939, 196.3474579, -256.2637939, 196.3474579, -452.6112671, 452.6112671
5: -229.6463013, 178.5066833, -229.6463013, 178.5066833, -408.1529541, 408.1529541
6: -219.7590637, 210.8946838, -219.7590637, 210.8946838, -430.6536865, 430.6536865
7: -239.1515045, 200.9265594, -239.1515045, 200.9265594, -440.0780334, 440.0780334
8: -288.1658630, 196.6365509, -288.1658630, 196.6365509, -484.8024292, 484.8024292
9: -217.7042542, 214.2737274, -217.7042542, 214.2737274, -431.9779663, 431.9779663

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2256723, upper bound: 461.2259196
time: 10.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2267667, upper bound: 461.2264992
time: 9.61 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -242.0842743, 191.5462341, -430.6205750, 431.2466736
1: -199.7922211, 167.9400635, -202.2670288, 170.0345459, -369.8267822, 370.2070923
2: -262.6058350, 170.4243469, -265.9084778, 172.5523376, -435.1581116, 436.3327942
3: -279.3811340, 147.4806366, -282.9065552, 149.2875214, -428.6686401, 430.3872070
4: -256.2637939, 196.3474579, -259.4692688, 198.7713928, -455.0351868, 455.8167114
5: -229.6463013, 178.5066833, -232.5094452, 180.7077179, -410.3540039, 411.0160217
6: -219.7590637, 210.8946838, -222.5233765, 213.5542908, -433.3133545, 433.4180603
7: -239.1515045, 200.9265594, -242.1613312, 203.4472351, -442.5986938, 443.0878296
8: -288.1658630, 196.6365509, -291.7500916, 199.0496979, -487.2155762, 488.3866577
9: -217.7042542, 214.2737274, -220.4269104, 216.9405365, -434.6447754, 434.7006226

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2256723, upper bound: 461.2259196
time: 9.18 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2267667, upper bound: 461.2264992
time: 8.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -242.0842743, 191.5462341, -239.0743866, 189.1624146, -431.2466736, 430.6205750
1: -202.2670288, 170.0345459, -199.7922211, 167.9400635, -370.2070923, 369.8267822
2: -265.9084778, 172.5523376, -262.6058350, 170.4243469, -436.3327942, 435.1581116
3: -282.9065552, 149.2875214, -279.3811340, 147.4806366, -430.3872070, 428.6686401
4: -259.4692688, 198.7713928, -256.2637939, 196.3474579, -455.8167114, 455.0351868
5: -232.5094452, 180.7077179, -229.6463013, 178.5066833, -411.0160217, 410.3540039
6: -222.5233765, 213.5542908, -219.7590637, 210.8946838, -433.4180603, 433.3133545
7: -242.1613312, 203.4472351, -239.1515045, 200.9265594, -443.0878296, 442.5986938
8: -291.7500916, 199.0496979, -288.1658630, 196.6365509, -488.3866577, 487.2155762
9: -220.4269104, 216.9405365, -217.7042542, 214.2737274, -434.7006226, 434.6447754

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2252989, upper bound: 461.2257069
time: 7.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2261024, upper bound: 461.2261024
time: 9.72 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -242.0842743, 191.5462341, -242.0842743, 191.5462341, -433.6304932, 433.6304932
1: -202.2670288, 170.0345459, -202.2670288, 170.0345459, -372.3015747, 372.3015747
2: -265.9084778, 172.5523376, -265.9084778, 172.5523376, -438.4607544, 438.4607544
3: -282.9065552, 149.2875214, -282.9065552, 149.2875214, -432.1940918, 432.1940918
4: -259.4692688, 198.7713928, -259.4692688, 198.7713928, -458.2406616, 458.2406616
5: -232.5094452, 180.7077179, -232.5094452, 180.7077179, -413.2171631, 413.2171631
6: -222.5233765, 213.5542908, -222.5233765, 213.5542908, -436.0776672, 436.0776672
7: -242.1613312, 203.4472351, -242.1613312, 203.4472351, -445.6084595, 445.6084595
8: -291.7500916, 199.0496979, -291.7500916, 199.0496979, -490.7998047, 490.7998047
9: -220.4269104, 216.9405365, -220.4269104, 216.9405365, -437.3674316, 437.3674316

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2252989, upper bound: 461.2257069
time: 9.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2261024, upper bound: 461.2261024
time: 9.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.22 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2256723, upper bound: 461.2259196
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2267667, upper bound: 461.2264992
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2256723, upper bound: 461.2259196
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2267667, upper bound: 461.2264992
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2252989, upper bound: 461.2257069
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2261024, upper bound: 461.2261024
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2252989, upper bound: 461.2257069
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -461.2261024, upper bound: 461.2261024

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -232.4235077, 183.9037628, -238.2551422, 188.5147552, -420.9382629, 422.1589050
1: -194.2172089, 163.2401276, -199.1062927, 167.3616180, -361.5788269, 362.3464355
2: -255.2803345, 165.6826782, -261.7037964, 169.8411102, -425.1213989, 427.3864136
3: -271.5780945, 143.4040833, -278.4203186, 146.9796143, -418.5577087, 421.8244019
4: -249.0393982, 190.8573151, -255.3746643, 195.6718597, -444.7112427, 446.2319641
5: -223.2463684, 173.4654694, -228.8589783, 177.8864746, -401.1328430, 402.3244629
6: -213.6559143, 205.0137634, -219.0073090, 210.1702271, -423.8261414, 424.0210571
7: -232.4221954, 195.3165131, -238.3231354, 200.2363129, -432.6585083, 433.6396179
8: -280.1938171, 191.1667938, -287.1835938, 195.9635315, -476.1573486, 478.3504028
9: -211.5676117, 208.2878265, -216.9494324, 213.5368652, -425.1044312, 425.2372131

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2223593, upper bound: 461.2229048
time: 9.59 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2225946, upper bound: 461.2234612
time: 9.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -239.0743866, 189.1624146, -424.4051819, 425.2056274
1: -196.5772400, 165.2260284, -199.7922211, 167.9400635, -364.5172729, 365.0182495
2: -258.3783569, 167.6993256, -262.6058350, 170.4243469, -428.8026733, 430.3051147
3: -274.8818665, 145.1331940, -279.3811340, 147.4806366, -422.3624878, 424.5142822
4: -252.0967102, 193.1814880, -256.2637939, 196.3474579, -448.4441528, 449.4452820
5: -225.9644165, 175.5948639, -229.6463013, 178.5066833, -404.4710388, 405.2411499
6: -216.2400818, 207.4966888, -219.7590637, 210.8946838, -427.1347351, 427.2557068
7: -235.2669067, 197.6948547, -239.1515045, 200.9265594, -436.1934509, 436.8463745
8: -283.5543823, 193.4777679, -288.1658630, 196.6365509, -480.1909180, 481.6436157
9: -214.1685791, 210.8134613, -217.7042542, 214.2737274, -428.4423218, 428.5177002

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2269326, upper bound: 461.2264600
time: 9.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2269326, upper bound: 461.2276338
time: 9.37 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -232.4235077, 183.9037628, -241.2835236, 190.9132538, -423.3367004, 425.1872864
1: -194.2172089, 163.2401276, -201.5966797, 169.4692993, -363.6865234, 364.8367920
2: -255.2803345, 165.6826782, -265.0269775, 171.9823151, -427.2626343, 430.7096252
3: -271.5780945, 143.4040833, -281.9676819, 148.7977905, -420.3758850, 425.3717651
4: -249.0393982, 190.8573151, -258.6007690, 198.1111755, -447.1505737, 449.4580688
5: -223.2463684, 173.4654694, -231.7401581, 180.1019592, -403.3483276, 405.2055664
6: -213.6559143, 205.0137634, -221.7886963, 212.8464813, -426.5023193, 426.8024597
7: -232.4221954, 195.3165131, -241.3520050, 202.7728424, -435.1950378, 436.6685181
8: -280.1938171, 191.1667938, -290.7901917, 198.3920898, -478.5859070, 481.9569702
9: -211.5676117, 208.2878265, -219.6899719, 216.2205658, -427.7881775, 427.9777832

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2215239, upper bound: 461.2216494
time: 9.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2216948, upper bound: 461.2221536
time: 9.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -242.0842743, 191.5462341, -426.7889709, 428.2155151
1: -196.5772400, 165.2260284, -202.2670288, 170.0345459, -366.6117859, 367.4930420
2: -258.3783569, 167.6993256, -265.9084778, 172.5523376, -430.9306641, 433.6077576
3: -274.8818665, 145.1331940, -282.9065552, 149.2875214, -424.1693726, 428.0397339
4: -252.0967102, 193.1814880, -259.4692688, 198.7713928, -450.8680725, 452.6507568
5: -225.9644165, 175.5948639, -232.5094452, 180.7077179, -406.6721191, 408.1043091
6: -216.2400818, 207.4966888, -222.5233765, 213.5542908, -429.7943420, 430.0200806
7: -235.2669067, 197.6948547, -242.1613312, 203.4472351, -438.7140808, 439.8561401
8: -283.5543823, 193.4777679, -291.7500916, 199.0496979, -482.6040649, 485.2278442
9: -214.1685791, 210.8134613, -220.4269104, 216.9405365, -431.1091309, 431.2403564

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2263523, upper bound: 461.2256687
time: 10.49 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2263523, upper bound: 461.2264992
time: 10.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -235.9754791, 186.7184753, -238.2551422, 188.5147552, -424.4902344, 424.9736328
1: -197.1470947, 165.7136841, -199.1062927, 167.3616180, -364.5087280, 364.8199768
2: -259.1856079, 168.1936035, -261.7037964, 169.8411102, -429.0266724, 429.8973389
3: -275.7452698, 145.5431213, -278.4203186, 146.9796143, -422.7248840, 423.9634399
4: -252.8399200, 193.7277832, -255.3746643, 195.6718597, -448.5117798, 449.1023865
5: -226.6326752, 176.0688324, -228.8589783, 177.8864746, -404.5191650, 404.9277954
6: -216.9250336, 208.1578827, -219.0073090, 210.1702271, -427.0952759, 427.1651611
7: -235.9821777, 198.2952118, -238.3231354, 200.2363129, -436.2185059, 436.6183167
8: -284.4442139, 194.0287933, -287.1835938, 195.9635315, -480.4077454, 481.2123718
9: -214.7938232, 211.4463043, -216.9494324, 213.5368652, -428.3306580, 428.3956909

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2214882, upper bound: 461.2221987
time: 9.88 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2217167, upper bound: 461.2227657
time: 9.90 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -238.1799164, 188.4585419, -239.0743866, 189.1624146, -427.3422546, 427.5328979
1: -198.9911957, 167.2688141, -199.7922211, 167.9400635, -366.9312744, 367.0610352
2: -261.6021729, 169.7745209, -262.6058350, 170.4243469, -432.0264893, 432.3803101
3: -278.3209229, 146.8946838, -279.3811340, 147.4806366, -425.8015747, 426.2758179
4: -255.2234039, 195.5442047, -256.2637939, 196.3474579, -451.5708618, 451.8079834
5: -228.7573853, 177.7393646, -229.6463013, 178.5066833, -407.2639771, 407.3856812
6: -218.9365540, 210.0917358, -219.7590637, 210.8946838, -429.8312378, 429.8507996
7: -238.1993561, 200.1532288, -239.1515045, 200.9265594, -439.1259155, 439.3047180
8: -287.0516663, 195.8308258, -288.1658630, 196.6365509, -483.6882324, 483.9966736
9: -216.8212128, 213.4137726, -217.7042542, 214.2737274, -431.0949402, 431.1180115

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259196, upper bound: 461.2256720
time: 11.03 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259196, upper bound: 461.2267667
time: 9.64 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -235.9754791, 186.7184753, -241.2835236, 190.9132538, -426.8887329, 428.0020142
1: -197.1470947, 165.7136841, -201.5966797, 169.4692993, -366.6163940, 367.3103333
2: -259.1856079, 168.1936035, -265.0269775, 171.9823151, -431.1679077, 433.2205505
3: -275.7452698, 145.5431213, -281.9676819, 148.7977905, -424.5430603, 427.5108032
4: -252.8399200, 193.7277832, -258.6007690, 198.1111755, -450.9511108, 452.3285522
5: -226.6326752, 176.0688324, -231.7401581, 180.1019592, -406.7346191, 407.8088989
6: -216.9250336, 208.1578827, -221.7886963, 212.8464813, -429.7714539, 429.9465637
7: -235.9821777, 198.2952118, -241.3520050, 202.7728424, -438.7550049, 439.6472168
8: -284.4442139, 194.0287933, -290.7901917, 198.3920898, -482.8363037, 484.8189697
9: -214.7938232, 211.4463043, -219.6899719, 216.2205658, -431.0144043, 431.1362610

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211403, upper bound: 461.2214018
time: 7.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2213269, upper bound: 461.2218852
time: 8.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -238.1799164, 188.4585419, -242.0842743, 191.5462341, -429.7260437, 430.5427856
1: -198.9911957, 167.2688141, -202.2670288, 170.0345459, -369.0257568, 369.5358276
2: -261.6021729, 169.7745209, -265.9084778, 172.5523376, -434.1544495, 435.6829529
3: -278.3209229, 146.8946838, -282.9065552, 149.2875214, -427.6084595, 429.8012390
4: -255.2234039, 195.5442047, -259.4692688, 198.7713928, -453.9948120, 455.0134888
5: -228.7573853, 177.7393646, -232.5094452, 180.7077179, -409.4650879, 410.2488098
6: -218.9365540, 210.0917358, -222.5233765, 213.5542908, -432.4908447, 432.6151123
7: -238.1993561, 200.1532288, -242.1613312, 203.4472351, -441.6465454, 442.3144836
8: -287.0516663, 195.8308258, -291.7500916, 199.0496979, -486.1013489, 487.5809326
9: -216.8212128, 213.4137726, -220.4269104, 216.9405365, -433.7617493, 433.8406677

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2257066, upper bound: 461.2252988
time: 9.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2257066, upper bound: 461.2261024
time: 9.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.44 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2223593, upper bound: 461.2229048
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2225946, upper bound: 461.2234612
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2269326, upper bound: 461.2264600
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2269326, upper bound: 461.2276338
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2215239, upper bound: 461.2216494
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2216948, upper bound: 461.2221536
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2263523, upper bound: 461.2256687
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2263523, upper bound: 461.2264992
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2214882, upper bound: 461.2221987
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2217167, upper bound: 461.2227657
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2259196, upper bound: 461.2256720
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2259196, upper bound: 461.2267667
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2211403, upper bound: 461.2214018
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2213269, upper bound: 461.2218852
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2257066, upper bound: 461.2252988
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.44
Output dim: 7, lower bound: -461.2257066, upper bound: 461.2261024

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -232.4235077, 183.9037628, -223.8698578, 177.1644897, -409.5880127, 407.7736206
1: -194.2172089, 163.2401276, -187.0092316, 157.1434937, -351.3606567, 350.2493591
2: -255.2803345, 165.6826782, -245.7913361, 159.5826111, -414.8629456, 411.4739685
3: -271.5780945, 143.4040833, -261.4521484, 137.9939728, -409.5720825, 404.8562317
4: -249.0393982, 190.8573151, -239.8807983, 183.8171692, -432.8565674, 430.7380981
5: -223.2463684, 173.4654694, -215.0943298, 167.1474609, -390.3938293, 388.5598145
6: -213.6559143, 205.0137634, -205.7490845, 197.3992310, -411.0550537, 410.7627869
7: -232.4221954, 195.3165131, -223.8441620, 188.1179962, -420.5401917, 419.1606750
8: -280.1938171, 191.1667938, -269.8129883, 184.1234741, -464.3172913, 460.9797974
9: -211.5676117, 208.2878265, -203.7278137, 200.5552826, -412.1228638, 412.0156250

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2219038, upper bound: 461.2228510
time: 8.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2219038, upper bound: 461.2229000
time: 10.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -231.5499725, 183.2148895, -229.5080872, 181.6374207, -413.1873779, 412.7229614
1: -193.4834747, 162.6215210, -191.7588043, 161.1511383, -354.6345520, 354.3803101
2: -254.3145447, 165.0616302, -252.0169678, 163.6055298, -417.9200745, 417.0786133
3: -270.5506897, 142.8601379, -268.1217957, 141.4983978, -412.0490723, 410.9819336
4: -248.0984802, 190.1382446, -245.9393311, 188.4525452, -436.5509949, 436.0775757
5: -222.4096222, 172.8153229, -220.5041809, 171.3946075, -393.8041992, 393.3194580
6: -212.8524628, 204.2385559, -210.9494019, 202.3905792, -415.2430420, 415.1879578
7: -231.5457001, 194.5835571, -229.5347595, 192.9014282, -424.4471130, 424.1183167
8: -279.1398926, 190.4474335, -276.6226807, 188.7286682, -467.8685608, 467.0701294
9: -210.7687378, 207.5017395, -208.9352417, 205.6385040, -416.4072266, 416.4369812

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2219601, upper bound: 461.2232921
time: 9.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2219601, upper bound: 461.2234612
time: 9.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -232.4235077, 183.9037628, -419.1465454, 418.5547180
1: -196.5772400, 165.2260284, -194.2172089, 163.2401276, -359.8173828, 359.4432373
2: -258.3783569, 167.6993256, -255.2803345, 165.6826782, -424.0610046, 422.9796143
3: -274.8818665, 145.1331940, -271.5780945, 143.4040833, -418.2859497, 416.7112732
4: -252.0967102, 193.1814880, -249.0393982, 190.8573151, -442.9540100, 442.2208862
5: -225.9644165, 175.5948639, -223.2463684, 173.4654694, -399.4298706, 398.8412476
6: -216.2400818, 207.4966888, -213.6559143, 205.0137634, -421.2538147, 421.1525269
7: -235.2669067, 197.6948547, -232.4221954, 195.3165131, -430.5834045, 430.1170654
8: -283.5543823, 193.4777679, -280.1938171, 191.1667938, -474.7211914, 473.6715698
9: -214.1685791, 210.8134613, -211.5676117, 208.2878265, -422.4564209, 422.3810425

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2229048, upper bound: 461.2223593
time: 10.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2234612, upper bound: 461.2225946
time: 9.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -235.2428131, 186.1312866, -421.3739929, 421.3739929
1: -196.5772400, 165.2260284, -196.5772400, 165.2260284, -361.8032227, 361.8032227
2: -258.3783569, 167.6993256, -258.3783569, 167.6993256, -426.0776367, 426.0776367
3: -274.8818665, 145.1331940, -274.8818665, 145.1331940, -420.0150452, 420.0150452
4: -252.0967102, 193.1814880, -252.0967102, 193.1814880, -445.2781372, 445.2781372
5: -225.9644165, 175.5948639, -225.9644165, 175.5948639, -401.5592651, 401.5592651
6: -216.2400818, 207.4966888, -216.2400818, 207.4966888, -423.7367554, 423.7367554
7: -235.2669067, 197.6948547, -235.2669067, 197.6948547, -432.9617615, 432.9617615
8: -283.5543823, 193.4777679, -283.5543823, 193.4777679, -477.0321655, 477.0321655
9: -214.1685791, 210.8134613, -214.1685791, 210.8134613, -424.9820557, 424.9820557

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2229048, upper bound: 461.2240812
time: 10.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2234612, upper bound: 461.2243082
time: 9.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -232.4235077, 183.9037628, -227.1416016, 179.7548523, -412.1783142, 411.0453491
1: -194.2172089, 163.2401276, -189.7092133, 159.4232788, -353.6405029, 352.9493408
2: -255.2803345, 165.6826782, -249.3856354, 161.8927307, -417.1730652, 415.0682678
3: -271.5780945, 143.4040833, -265.2897034, 139.9585876, -411.5366821, 408.6937866
4: -249.0393982, 190.8573151, -243.3728790, 186.4596710, -435.4990845, 434.2301941
5: -223.2463684, 173.4654694, -218.2075806, 169.5443115, -392.7906799, 391.6730347
6: -213.6559143, 205.0137634, -208.7578430, 200.2935181, -413.9494019, 413.7715759
7: -232.4221954, 195.3165131, -227.1205750, 190.8579254, -423.2801208, 422.4370728
8: -280.1938171, 191.1667938, -273.7145996, 186.7485046, -466.9422913, 464.8814087
9: -211.5676117, 208.2878265, -206.6919861, 203.4618073, -415.0293579, 414.9797974

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2210809, upper bound: 461.2215847
time: 11.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2210809, upper bound: 461.2216326
time: 11.10 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -231.5499725, 183.2148895, -231.9918976, 183.6101990, -415.1601562, 415.2067871
1: -193.4834747, 162.6215210, -193.7906647, 162.8722687, -356.3557129, 356.4121704
2: -254.3145447, 165.0616302, -254.7357788, 165.3552399, -419.6697388, 419.7974243
3: -270.5506897, 142.8601379, -271.0251770, 142.9804077, -413.5310974, 413.8853149
4: -248.0984802, 190.1382446, -248.5791931, 190.4430237, -438.5415039, 438.7174072
5: -222.4096222, 172.8153229, -222.8633575, 173.2004700, -395.6100159, 395.6786804
6: -212.8524628, 204.2385559, -213.2278137, 204.5824432, -417.4349060, 417.4663696
7: -231.5457001, 194.5835571, -232.0126495, 194.9751282, -426.5208130, 426.5961914
8: -279.1398926, 190.4474335, -279.5686340, 190.7138824, -469.8537292, 470.0160522
9: -210.7687378, 207.5017395, -211.1723633, 207.8327484, -418.6015015, 418.6740723

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211053, upper bound: 461.2219622
time: 9.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211053, upper bound: 461.2221536
time: 10.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -235.9754791, 186.7184753, -421.9612427, 422.1067505
1: -196.5772400, 165.2260284, -197.1470947, 165.7136841, -362.2909241, 362.3731079
2: -258.3783569, 167.6993256, -259.1856079, 168.1936035, -426.5719299, 426.8848877
3: -274.8818665, 145.1331940, -275.7452698, 145.5431213, -420.4249878, 420.8784790
4: -252.0967102, 193.1814880, -252.8399200, 193.7277832, -445.8244324, 446.0214233
5: -225.9644165, 175.5948639, -226.6326752, 176.0688324, -402.0332642, 402.2275391
6: -216.2400818, 207.4966888, -216.9250336, 208.1578827, -424.3978882, 424.4216919
7: -235.2669067, 197.6948547, -235.9821777, 198.2952118, -433.5620728, 433.6770325
8: -283.5543823, 193.4777679, -284.4442139, 194.0287933, -477.5831909, 477.9219971
9: -214.1685791, 210.8134613, -214.7938232, 211.4463043, -425.6148682, 425.6072693

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2221987, upper bound: 461.2214882
time: 9.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2227657, upper bound: 461.2217167
time: 10.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -238.1799164, 188.4585419, -423.7012634, 424.3110962
1: -196.5772400, 165.2260284, -198.9911957, 167.2688141, -363.8460083, 364.2172241
2: -258.3783569, 167.6993256, -261.6021729, 169.7745209, -428.1528625, 429.3014526
3: -274.8818665, 145.1331940, -278.3209229, 146.8946838, -421.7765503, 423.4541016
4: -252.0967102, 193.1814880, -255.2234039, 195.5442047, -447.6408691, 448.4049072
5: -225.9644165, 175.5948639, -228.7573853, 177.7393646, -403.7037964, 404.3522339
6: -216.2400818, 207.4966888, -218.9365540, 210.0917358, -426.3318176, 426.4332275
7: -235.2669067, 197.6948547, -238.1993561, 200.1532288, -435.4201050, 435.8942261
8: -283.5543823, 193.4777679, -287.0516663, 195.8308258, -479.3851929, 480.5294189
9: -214.1685791, 210.8134613, -216.8212128, 213.4137726, -427.5823364, 427.6346436

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2221987, upper bound: 461.2226221
time: 9.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2227657, upper bound: 461.2228253
time: 9.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -235.9754791, 186.7184753, -223.8698578, 177.1644897, -413.1399536, 410.5883179
1: -197.1470947, 165.7136841, -187.0092316, 157.1434937, -354.2905884, 352.7229004
2: -259.1856079, 168.1936035, -245.7913361, 159.5826111, -418.7682190, 413.9848938
3: -275.7452698, 145.5431213, -261.4521484, 137.9939728, -413.7392578, 406.9952698
4: -252.8399200, 193.7277832, -239.8807983, 183.8171692, -436.6571045, 433.6085510
5: -226.6326752, 176.0688324, -215.0943298, 167.1474609, -393.7801514, 391.1631470
6: -216.9250336, 208.1578827, -205.7490845, 197.3992310, -414.3242188, 413.9068909
7: -235.9821777, 198.2952118, -223.8441620, 188.1179962, -424.1001587, 422.1393738
8: -284.4442139, 194.0287933, -269.8129883, 184.1234741, -468.5676880, 463.8417969
9: -214.7938232, 211.4463043, -203.7278137, 200.5552826, -415.3490906, 415.1741028

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2210835, upper bound: 461.2221527
time: 10.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2210835, upper bound: 461.2221977
time: 9.51 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -235.0864716, 186.0174103, -229.5080872, 181.6374207, -416.7238770, 415.5255127
1: -196.3999634, 165.0839539, -191.7588043, 161.1511383, -357.5510864, 356.8427734
2: -258.2024536, 167.5617523, -252.0169678, 163.6055298, -421.8079224, 419.5787048
3: -274.6992493, 144.9893799, -268.1217957, 141.4983978, -416.1976318, 413.1111755
4: -251.8825378, 192.9958038, -245.9393311, 188.4525452, -440.3350830, 438.9351196
5: -225.7807007, 175.4067383, -220.5041809, 171.3946075, -397.1752930, 395.9109192
6: -216.1073608, 207.3688965, -210.9494019, 202.3905792, -418.4979248, 418.3182373
7: -235.0903473, 197.5488892, -229.5347595, 192.9014282, -427.9917297, 427.0836487
8: -283.3717651, 193.2968597, -276.6226807, 188.7286682, -472.1004333, 469.9195557
9: -213.9804535, 210.6463013, -208.9352417, 205.6385040, -419.6189575, 419.5815430

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211418, upper bound: 461.2225940
time: 9.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211418, upper bound: 461.2227657
time: 9.50 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -238.1799164, 188.4585419, -232.4235077, 183.9037628, -422.0836487, 420.8819885
1: -198.9911957, 167.2688141, -194.2172089, 163.2401276, -362.2313232, 361.4859924
2: -261.6021729, 169.7745209, -255.2803345, 165.6826782, -427.2848206, 425.0548096
3: -278.3209229, 146.8946838, -271.5780945, 143.4040833, -421.7250061, 418.4727783
4: -255.2234039, 195.5442047, -249.0393982, 190.8573151, -446.0807190, 444.5836182
5: -228.7573853, 177.7393646, -223.2463684, 173.4654694, -402.2228394, 400.9857178
6: -218.9365540, 210.0917358, -213.6559143, 205.0137634, -423.9503174, 423.7476501
7: -238.1993561, 200.1532288, -232.4221954, 195.3165131, -433.5158691, 432.5754395
8: -287.0516663, 195.8308258, -280.1938171, 191.1667938, -478.2184448, 476.0246582
9: -216.8212128, 213.4137726, -211.5676117, 208.2878265, -425.1090088, 424.9813538

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2216494, upper bound: 461.2215239
time: 9.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2221536, upper bound: 461.2216948
time: 9.24 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -238.1799164, 188.4585419, -235.2428131, 186.1312866, -424.3110962, 423.7012634
1: -198.9911957, 167.2688141, -196.5772400, 165.2260284, -364.2172241, 363.8460083
2: -261.6021729, 169.7745209, -258.3783569, 167.6993256, -429.3014526, 428.1528625
3: -278.3209229, 146.8946838, -274.8818665, 145.1331940, -423.4541016, 421.7765503
4: -255.2234039, 195.5442047, -252.0967102, 193.1814880, -448.4049072, 447.6408691
5: -228.7573853, 177.7393646, -225.9644165, 175.5948639, -404.3522339, 403.7037964
6: -218.9365540, 210.0917358, -216.2400818, 207.4966888, -426.4332275, 426.3318176
7: -238.1993561, 200.1532288, -235.2669067, 197.6948547, -435.8942261, 435.4201050
8: -287.0516663, 195.8308258, -283.5543823, 193.4777679, -480.5294189, 479.3851929
9: -216.8212128, 213.4137726, -214.1685791, 210.8134613, -427.6346436, 427.5823364

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2216494, upper bound: 461.2229661
time: 9.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2221536, upper bound: 461.2230895
time: 10.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -235.9754791, 186.7184753, -227.1416016, 179.7548523, -415.7303467, 413.8600769
1: -197.1470947, 165.7136841, -189.7092133, 159.4232788, -356.5703735, 355.4229126
2: -259.1856079, 168.1936035, -249.3856354, 161.8927307, -421.0783386, 417.5791931
3: -275.7452698, 145.5431213, -265.2897034, 139.9585876, -415.7038574, 410.8328247
4: -252.8399200, 193.7277832, -243.3728790, 186.4596710, -439.2995911, 437.1006470
5: -226.6326752, 176.0688324, -218.2075806, 169.5443115, -396.1770020, 394.2763977
6: -216.9250336, 208.1578827, -208.7578430, 200.2935181, -417.2185364, 416.9156799
7: -235.9821777, 198.2952118, -227.1205750, 190.8579254, -426.8400879, 425.4157410
8: -284.4442139, 194.0287933, -273.7145996, 186.7485046, -471.1926880, 467.7434082
9: -214.7938232, 211.4463043, -206.6919861, 203.4618073, -418.2556152, 418.1382751

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2207062, upper bound: 461.2213331
time: 9.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2207062, upper bound: 461.2213846
time: 9.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -235.0864716, 186.0174103, -231.9918976, 183.6101990, -418.6966553, 418.0093079
1: -196.3999634, 165.0839539, -193.7906647, 162.8722687, -359.2722168, 358.8746338
2: -258.2024536, 167.5617523, -254.7357788, 165.3552399, -423.5576172, 422.2974243
3: -274.6992493, 144.9893799, -271.0251770, 142.9804077, -417.6796570, 416.0145569
4: -251.8825378, 192.9958038, -248.5791931, 190.4430237, -442.3255615, 441.5749817
5: -225.7807007, 175.4067383, -222.8633575, 173.2004700, -398.9811401, 398.2700806
6: -216.1073608, 207.3688965, -213.2278137, 204.5824432, -420.6898193, 420.5966797
7: -235.0903473, 197.5488892, -232.0126495, 194.9751282, -430.0654907, 429.5615234
8: -283.3717651, 193.2968597, -279.5686340, 190.7138824, -474.0856323, 472.8654785
9: -213.9804535, 210.6463013, -211.1723633, 207.8327484, -421.8131714, 421.8186035

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2207472, upper bound: 461.2217144
time: 9.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2207472, upper bound: 461.2218852
time: 8.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.54 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2219038, upper bound: 461.2228510
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2219038, upper bound: 461.2229000
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2219601, upper bound: 461.2232921
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2219601, upper bound: 461.2234612
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2229048, upper bound: 461.2223593
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2234612, upper bound: 461.2225946
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2229048, upper bound: 461.2240812
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2234612, upper bound: 461.2243082
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2210809, upper bound: 461.2215847
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2210809, upper bound: 461.2216326
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2211053, upper bound: 461.2219622
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2211053, upper bound: 461.2221536
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2221987, upper bound: 461.2214882
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2227657, upper bound: 461.2217167
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2221987, upper bound: 461.2226221
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2227657, upper bound: 461.2228253
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2210835, upper bound: 461.2221527
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2210835, upper bound: 461.2221977
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2211418, upper bound: 461.2225940
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2211418, upper bound: 461.2227657
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2216494, upper bound: 461.2215239
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2221536, upper bound: 461.2216948
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2216494, upper bound: 461.2229661
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2221536, upper bound: 461.2230895
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2207062, upper bound: 461.2213331
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2207062, upper bound: 461.2213846
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2207472, upper bound: 461.2217144
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.54
Output dim: 7, lower bound: -461.2207472, upper bound: 461.2218852
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.54
Output dim: 7, lower bound: -461.2257066, upper bound: 461.2252988
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.54
Output dim: 7, lower bound: -461.2257066, upper bound: 461.2261024
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=464.3514404296875
rel_dist={7: [-461.2313236619183, 461.23132366191817]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2273437, upper bound: 461.2271808
time: 10.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2269749, upper bound: 461.2269749
time: 11.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.66
Output dim: 7, lower bound: -461.2273437, upper bound: 461.2271808
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.66
Output dim: 7, lower bound: -461.2269749, upper bound: 461.2269749

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -249.3726196, 197.2948303, -436.3692017, 438.5350037
1: -199.7922211, 167.9400635, -208.4724426, 175.2137146, -375.0059204, 376.4124756
2: -262.6058350, 170.4243469, -273.9754028, 177.7326660, -440.3385010, 444.3997192
3: -279.3811340, 147.4806366, -291.4800110, 153.8207550, -433.2018738, 438.9606323
4: -256.2637939, 196.3474579, -267.3577881, 204.8517761, -461.1155701, 463.7052612
5: -229.6463013, 178.5066833, -239.5769196, 186.2265320, -415.8728333, 418.0835266
6: -219.7590637, 210.8946838, -229.2321167, 220.0014038, -439.7604370, 440.1267700
7: -239.1515045, 200.9265594, -249.4992981, 209.5615845, -448.7130737, 450.4258118
8: -288.1658630, 196.6365509, -300.5774841, 205.0891724, -493.2550354, 497.2140503
9: -217.7042542, 214.2737274, -227.0950012, 223.5180817, -441.2223511, 441.3687134

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2269749, upper bound: 461.2269749
time: 10.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2269749, upper bound: 461.2269749
time: 9.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -242.0842743, 191.5462341, -244.7890625, 193.6768951, -435.7611694, 436.3352661
1: -202.2670288, 170.0345459, -204.6220245, 171.9915161, -374.2585449, 374.6565552
2: -265.9084778, 172.5523376, -268.9311829, 174.4924774, -440.4008789, 441.4835205
3: -282.9065552, 149.2875214, -286.1080322, 151.0102234, -433.9167786, 435.3955688
4: -259.4692688, 198.7713928, -262.4234009, 201.0766296, -460.5458984, 461.1947937
5: -232.5094452, 180.7077179, -235.1615601, 182.8023682, -415.3117981, 415.8692627
6: -222.5233765, 213.5542908, -225.0197601, 215.9559937, -438.4793701, 438.5740356
7: -242.1613312, 203.4472351, -244.9081116, 205.7283630, -447.8896179, 448.3553162
8: -291.7500916, 199.0496979, -295.0585632, 201.3242188, -493.0743103, 494.1082764
9: -220.4269104, 216.9405365, -222.9252014, 219.4095001, -439.8363953, 439.8657227

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2253602, upper bound: 461.2250518
time: 10.65 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259029, upper bound: 461.2259029
time: 9.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.67 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.67
Output dim: 7, lower bound: -461.2269749, upper bound: 461.2269749
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.67
Output dim: 7, lower bound: -461.2269749, upper bound: 461.2269749
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.67
Output dim: 7, lower bound: -461.2253602, upper bound: 461.2250518
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.67
Output dim: 7, lower bound: -461.2259029, upper bound: 461.2259029

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -239.0743866, 189.1624146, -428.2367859, 428.2367859
1: -199.7922211, 167.9400635, -199.7922211, 167.9400635, -367.7322998, 367.7322998
2: -262.6058350, 170.4243469, -262.6058350, 170.4243469, -433.0301514, 433.0301514
3: -279.3811340, 147.4806366, -279.3811340, 147.4806366, -426.8617554, 426.8617554
4: -256.2637939, 196.3474579, -256.2637939, 196.3474579, -452.6112671, 452.6112671
5: -229.6463013, 178.5066833, -229.6463013, 178.5066833, -408.1529541, 408.1529541
6: -219.7590637, 210.8946838, -219.7590637, 210.8946838, -430.6536865, 430.6536865
7: -239.1515045, 200.9265594, -239.1515045, 200.9265594, -440.0780334, 440.0780334
8: -288.1658630, 196.6365509, -288.1658630, 196.6365509, -484.8024292, 484.8024292
9: -217.7042542, 214.2737274, -217.7042542, 214.2737274, -431.9779663, 431.9779663

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2252751, upper bound: 461.2254875
time: 10.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2263289, upper bound: 461.2261194
time: 10.33 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -242.0842743, 191.5462341, -430.6205750, 431.2466736
1: -199.7922211, 167.9400635, -202.2670288, 170.0345459, -369.8267822, 370.2070923
2: -262.6058350, 170.4243469, -265.9084778, 172.5523376, -435.1581116, 436.3327942
3: -279.3811340, 147.4806366, -282.9065552, 149.2875214, -428.6686401, 430.3872070
4: -256.2637939, 196.3474579, -259.4692688, 198.7713928, -455.0351868, 455.8167114
5: -229.6463013, 178.5066833, -232.5094452, 180.7077179, -410.3540039, 411.0160217
6: -219.7590637, 210.8946838, -222.5233765, 213.5542908, -433.3133545, 433.4180603
7: -239.1515045, 200.9265594, -242.1613312, 203.4472351, -442.5986938, 443.0878296
8: -288.1658630, 196.6365509, -291.7500916, 199.0496979, -487.2155762, 488.3866577
9: -217.7042542, 214.2737274, -220.4269104, 216.9405365, -434.6447754, 434.7006226

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2252751, upper bound: 461.2254875
time: 10.19 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2263289, upper bound: 461.2261194
time: 10.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -239.2501373, 189.3060150, -238.1734619, 188.4454346, -427.6955566, 427.4794922
1: -199.8944550, 168.0341034, -199.0750122, 167.3154602, -367.2098999, 367.1090393
2: -262.7886963, 170.5349121, -261.6459045, 169.7739258, -432.5626221, 432.1808167
3: -279.5839539, 147.5543365, -278.3480225, 146.9543457, -426.5382996, 425.9023438
4: -256.3955383, 196.4347382, -255.2387238, 195.6146393, -452.0101624, 451.6734314
5: -229.7870483, 178.5639343, -228.7950439, 177.7892609, -407.5762634, 407.3589783
6: -219.9230499, 211.0491943, -218.9470520, 210.1072998, -430.0303345, 429.9961853
7: -239.2972107, 201.0605621, -238.2156830, 200.1473389, -439.4444885, 439.2762451
8: -288.3531494, 196.7223206, -287.1285095, 195.8838806, -484.2370300, 483.8508301
9: -217.8188934, 214.3927460, -216.8223114, 213.4559784, -431.2747803, 431.2150574

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2208798, upper bound: 461.2206829
time: 10.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2208782, upper bound: 461.2210250
time: 11.93 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -240.8780060, 190.5921478, -240.8768463, 190.5835724, -431.4615784, 431.4689941
1: -201.2548981, 169.1801910, -201.3386841, 169.2198334, -370.4746704, 370.5188599
2: -264.5779724, 171.6940460, -264.6153259, 171.7085876, -436.2865601, 436.3093262
3: -281.4896545, 148.5482788, -281.5134277, 148.6126556, -430.1022949, 430.0617065
4: -258.1575012, 197.7744141, -258.1689453, 197.8439026, -456.0014038, 455.9432983
5: -231.3500366, 179.7906799, -231.4026642, 179.8289642, -411.1789856, 411.1932678
6: -221.4152222, 212.4845123, -221.4266815, 212.4860382, -433.9012451, 433.9111633
7: -240.9372864, 202.4296112, -240.9396057, 202.4275208, -443.3647766, 443.3692017
8: -290.2985229, 198.0552368, -290.3493652, 198.0992432, -488.3977661, 488.4046021
9: -219.3128967, 215.8509674, -219.3126221, 215.8753815, -435.1882935, 435.1635742

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2217144, upper bound: 461.2219185
time: 9.88 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2221737, upper bound: 461.2221737
time: 9.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2252751, upper bound: 461.2254875
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2263289, upper bound: 461.2261194
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2252751, upper bound: 461.2254875
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2263289, upper bound: 461.2261194
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2208798, upper bound: 461.2206829
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2208782, upper bound: 461.2210250
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2217144, upper bound: 461.2219185
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.38
Output dim: 7, lower bound: -461.2221737, upper bound: 461.2221737

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -232.4235077, 183.9037628, -236.1734314, 186.8691254, -419.2926025, 420.0772095
1: -194.2172089, 163.2401276, -197.3631439, 165.8919220, -360.1091309, 360.6032715
2: -255.2803345, 165.6826782, -259.4115906, 168.3591766, -423.6394958, 425.0942383
3: -271.5780945, 143.4040833, -275.9789124, 145.7063293, -417.2843933, 419.3829956
4: -249.0393982, 190.8573151, -253.1154785, 193.9551239, -442.9944763, 443.9727783
5: -223.2463684, 173.4654694, -226.8585815, 176.3107910, -399.5571594, 400.3240356
6: -213.6559143, 205.0137634, -217.0971375, 208.3294983, -421.9853821, 422.1109009
7: -232.4221954, 195.3165131, -236.2183228, 198.4824524, -430.9046326, 431.5348206
8: -280.1938171, 191.1667938, -284.6876221, 194.2534180, -474.4472351, 475.8544312
9: -211.5676117, 208.2878265, -215.0317383, 211.6645966, -423.2321777, 423.3195801

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2218877, upper bound: 461.2221779
time: 9.73 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2222518, upper bound: 461.2227515
time: 12.33 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -237.8900757, 188.2255249, -423.4682922, 424.0213013
1: -196.5772400, 165.2260284, -198.7985229, 167.1013031, -363.6784973, 364.0245056
2: -258.3783569, 167.6993256, -261.2991638, 169.5821533, -427.9604492, 428.9984436
3: -274.8818665, 145.1331940, -277.9903259, 146.7551117, -421.6369629, 423.1235046
4: -252.0967102, 193.1814880, -254.9760590, 195.3689117, -447.4655457, 448.1575317
5: -225.9644165, 175.5948639, -228.5082550, 177.6066589, -403.5710449, 404.1031189
6: -216.2400818, 207.4966888, -218.6714020, 209.8444672, -426.0845337, 426.1680603
7: -235.2669067, 197.6948547, -237.9508514, 199.9276733, -435.1945496, 435.6456909
8: -283.5543823, 193.4777679, -286.7407227, 195.6603088, -479.2146912, 480.2185059
9: -214.1685791, 210.8134613, -216.6113434, 213.2043152, -427.3728943, 427.4247437

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2237369, upper bound: 461.2235097
time: 10.09 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2240525, upper bound: 461.2240525
time: 10.14 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -232.4235077, 183.9037628, -239.2501373, 189.3060150, -421.7295227, 423.1539001
1: -194.2172089, 163.2401276, -199.8944550, 168.0341034, -362.2512817, 363.1345825
2: -255.2803345, 165.6826782, -262.7886963, 170.5349121, -425.8152466, 428.4713440
3: -271.5780945, 143.4040833, -279.5839539, 147.5543365, -419.1324463, 422.9880371
4: -249.0393982, 190.8573151, -256.3955383, 196.4347382, -445.4741211, 447.2528687
5: -223.2463684, 173.4654694, -229.7870483, 178.5639343, -401.8103027, 403.2525024
6: -213.6559143, 205.0137634, -219.9230499, 211.0491943, -424.7050781, 424.9367981
7: -232.4221954, 195.3165131, -239.2972107, 201.0605621, -433.4827576, 434.6136780
8: -280.1938171, 191.1667938, -288.3531494, 196.7223206, -476.9161377, 479.5199585
9: -211.5676117, 208.2878265, -217.8188934, 214.3927460, -425.9603271, 426.1066589

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2209060, upper bound: 461.2209824
time: 9.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2212184, upper bound: 461.2214951
time: 11.58 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -235.2428131, 186.1312866, -240.8780060, 190.5921478, -425.8349304, 427.0092773
1: -196.5772400, 165.2260284, -201.2548981, 169.1801910, -365.7573853, 366.4808960
2: -258.3783569, 167.6993256, -264.5779724, 171.6940460, -430.0723877, 432.2772522
3: -274.8818665, 145.1331940, -281.4896545, 148.5482788, -423.4301453, 426.6228638
4: -252.0967102, 193.1814880, -258.1575012, 197.7744141, -449.8710327, 451.3389893
5: -225.9644165, 175.5948639, -231.3500366, 179.7906799, -405.7550659, 406.9448853
6: -216.2400818, 207.4966888, -221.4152222, 212.4845123, -428.7245789, 428.9119263
7: -235.2669067, 197.6948547, -240.9372864, 202.4296112, -437.6965332, 438.6321411
8: -283.5543823, 193.4777679, -290.2985229, 198.0552368, -481.6096191, 483.7763062
9: -214.1685791, 210.8134613, -219.3128967, 215.8509674, -430.0195312, 430.1263428

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2224402, upper bound: 461.2219304
time: 11.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2226641, upper bound: 461.2223814
time: 9.96 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -225.1155090, 178.1533508, -232.9854736, 184.3511505, -409.4666748, 411.1388245
1: -188.0131226, 157.9938660, -194.7138062, 163.6305084, -351.6436157, 352.7076416
2: -247.1558075, 160.4508514, -255.9076233, 166.0741730, -413.2299805, 416.3584595
3: -262.9147339, 138.7195892, -272.2289429, 143.7127991, -406.6275330, 410.9485474
4: -241.1761169, 184.7895355, -249.6523590, 191.3404083, -432.5165405, 434.4418945
5: -216.2615051, 168.0119629, -223.8293610, 173.9162445, -390.1777344, 391.8413086
6: -206.8992767, 198.5030060, -214.1663361, 205.5018616, -412.4011230, 412.6693420
7: -225.0730286, 189.1523895, -232.9946442, 195.7765656, -420.8495789, 422.1470032
8: -271.2874451, 185.0843811, -280.8653564, 191.6150818, -462.9025269, 465.9497375
9: -204.8277740, 201.6409912, -212.0536804, 208.7758331, -413.6036072, 413.6946716

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2176886, upper bound: 461.2177597
time: 11.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2174868, upper bound: 461.2174324
time: 11.94 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -229.9343109, 181.9852142, -233.3605042, 184.6510468, -414.5852661, 415.3457031
1: -192.0685425, 161.4212341, -195.0326843, 163.9080505, -355.9765930, 356.4539185
2: -252.4712219, 163.8916473, -256.3257751, 166.3538513, -418.8250122, 420.2174072
3: -268.6133728, 141.7223206, -272.6868896, 143.9581299, -412.5714722, 414.4092102
4: -246.3487091, 188.7471924, -250.0565033, 191.6535645, -438.0022583, 438.8037109
5: -220.8875427, 171.6459656, -224.1840973, 174.2070770, -395.0946045, 395.8300781
6: -211.3398895, 202.7641602, -214.5214996, 205.8364716, -417.1763000, 417.2856140
7: -229.9342041, 193.2438812, -233.3878632, 196.1096344, -426.0437317, 426.6317444
8: -277.1035156, 189.0254211, -281.3227844, 191.9211121, -469.0246277, 470.3481750
9: -209.2802887, 205.9844513, -212.4220428, 209.1265564, -418.4067688, 418.4064941

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2176886, upper bound: 461.2182588
time: 10.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2182647, upper bound: 461.2179053
time: 12.41 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -226.7339325, 179.4323120, -235.6954803, 186.4953461, -413.2292786, 415.1277161
1: -189.3660431, 159.1321716, -196.9841919, 165.5399475, -354.9059448, 356.1163635
2: -248.9341431, 161.6028900, -258.8859253, 168.0135345, -416.9476929, 420.4888000
3: -264.8088989, 139.7077332, -275.4024963, 145.3762207, -410.1850891, 415.1102295
4: -242.9266815, 186.1210632, -252.5900116, 193.5756226, -436.5023193, 438.7110291
5: -217.8155823, 169.2313385, -226.4445190, 175.9616699, -393.7772522, 395.6758423
6: -208.3821716, 199.9295349, -216.6517334, 207.8873749, -416.2695312, 416.5812683
7: -226.7037506, 190.5126648, -235.7251129, 198.0628815, -424.7666321, 426.2377930
8: -273.2202759, 186.4102783, -284.0943909, 193.8361053, -467.0563965, 470.5046692
9: -206.3128510, 203.0901489, -214.5495911, 211.2014465, -417.5142822, 417.6397400

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2184726, upper bound: 461.2189738
time: 11.02 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2182551, upper bound: 461.2185814
time: 11.88 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -231.5914154, 183.2929382, -236.0792236, 186.8001556, -418.3915710, 419.3721619
1: -193.4535675, 162.5864258, -197.3085022, 165.8218842, -359.2753601, 359.8949280
2: -254.2921906, 165.0703278, -259.3108215, 168.2971191, -422.5892944, 424.3811340
3: -270.5528564, 142.7340393, -275.8686829, 145.6243439, -416.1771851, 418.6027222
4: -248.1411438, 190.1102448, -253.0028687, 193.8940277, -442.0351562, 443.1130981
5: -222.4782104, 172.8926392, -226.8049469, 176.2570343, -398.7351990, 399.6975708
6: -212.8588867, 204.2248535, -217.0129242, 208.2282257, -421.0870972, 421.2377930
7: -231.6026306, 194.6356659, -236.1259155, 198.4000549, -430.0026245, 430.7615967
8: -279.0832214, 190.3811035, -284.5605774, 194.1486511, -473.2318726, 474.9416809
9: -210.7991791, 207.4671631, -214.9239502, 211.5584412, -422.3576050, 422.3911133

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2192724, upper bound: 461.2194535
time: 10.24 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2191007, upper bound: 461.2191007
time: 10.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.94 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2218877, upper bound: 461.2221779
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2222518, upper bound: 461.2227515
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2237369, upper bound: 461.2235097
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2240525, upper bound: 461.2240525
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2209060, upper bound: 461.2209824
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2212184, upper bound: 461.2214951
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2224402, upper bound: 461.2219304
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2226641, upper bound: 461.2223814
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2176886, upper bound: 461.2177597
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2174868, upper bound: 461.2174324
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2176886, upper bound: 461.2182588
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2182647, upper bound: 461.2179053
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2184726, upper bound: 461.2189738
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2182551, upper bound: 461.2185814
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2192724, upper bound: 461.2194535
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 7, lower bound: -461.2191007, upper bound: 461.2191007

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -227.1806793, 179.7659607, -221.7969513, 175.5261688, -402.7067871, 401.5629272
1: -189.8097687, 159.5156708, -185.2738342, 155.6804199, -345.4901428, 344.7894897
2: -249.4809570, 161.9442902, -243.5093384, 158.1074677, -407.5884094, 405.4535828
3: -265.3932800, 140.1294708, -259.0216980, 136.7264404, -402.1197205, 399.1511536
4: -243.3945160, 186.5373688, -237.6323853, 182.1079865, -425.5025024, 424.1697388
5: -218.2275238, 169.5514374, -213.1027832, 165.5787354, -383.8062439, 382.6542358
6: -208.8237457, 200.3593903, -203.8471527, 195.5668793, -404.3906250, 404.2065430
7: -227.1459503, 190.8996887, -221.7489166, 186.3721161, -413.5180359, 412.6486206
8: -273.8646851, 186.8535767, -267.3290405, 182.4210358, -456.2857056, 454.1826172
9: -206.7483673, 203.5579376, -201.8192444, 198.6914368, -405.4397583, 405.3771057

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2188886, upper bound: 461.2189386
time: 9.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2184925, upper bound: 461.2187722
time: 10.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -227.6767426, 180.1611938, -227.4318695, 179.9966278, -407.6733398, 407.5930786
1: -190.2305145, 159.8802032, -190.0206146, 159.6858673, -349.9163513, 349.9007568
2: -250.0333252, 162.3100586, -249.7314911, 162.1276550, -412.1608582, 412.0415649
3: -265.9970703, 140.4500122, -265.6877441, 140.2286530, -406.2257080, 406.1377563
4: -243.9278259, 186.9510956, -243.6862030, 186.7409363, -430.6687012, 430.6372986
5: -218.6991730, 169.9337616, -218.5099182, 169.8239899, -388.5231628, 388.4436646
6: -209.2910156, 200.8025665, -209.0441895, 200.5551605, -409.8461914, 409.8467407
7: -227.6615448, 191.3355560, -227.4356384, 191.1530762, -418.8145752, 418.7711792
8: -274.4692078, 187.2574463, -274.1344299, 187.0236359, -461.4928589, 461.3918457
9: -207.2283020, 204.0184174, -207.0237732, 203.7721710, -411.0004883, 411.0421753

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2194552, upper bound: 461.2198724
time: 10.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2191131, upper bound: 461.2197084
time: 9.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -230.0101776, 182.0015411, -223.5039978, 176.8750153, -406.8851929, 405.5055542
1: -192.1782837, 161.5088654, -186.7013092, 156.8825531, -349.0607605, 348.2101440
2: -252.5905914, 163.9678345, -245.3864899, 159.3226929, -411.9132690, 409.3543091
3: -268.7095032, 141.8653107, -261.0217285, 137.7692413, -406.4787598, 402.8870239
4: -246.4613647, 188.8697510, -239.4810181, 183.5138702, -429.9752197, 428.3507690
5: -220.9557953, 171.6885986, -214.7431488, 166.8671265, -387.8228760, 386.4317627
6: -211.4167938, 202.8510284, -205.4126434, 197.0727844, -408.4895630, 408.2636719
7: -230.0000305, 193.2866364, -223.4712524, 187.8088684, -417.8088989, 416.7578735
8: -277.2365417, 189.1719971, -269.3694763, 183.8196564, -461.0562134, 458.5414734
9: -209.3582916, 206.0918884, -203.3886261, 200.2223206, -409.5806274, 409.4805298

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2209000, upper bound: 461.2203800
time: 10.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2184925, upper bound: 461.2201866
time: 11.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -230.5041656, 182.3955078, -229.1485901, 181.3528595, -411.8570251, 411.5440979
1: -192.5975494, 161.8716888, -191.4562073, 160.8946533, -353.4921875, 353.3278809
2: -253.1396637, 164.3320007, -251.6187592, 163.3498230, -416.4895020, 415.9507141
3: -269.3107910, 142.1840515, -267.6984253, 141.2772827, -410.5880737, 409.8824768
4: -246.9949799, 189.2815247, -245.5469055, 188.1542206, -435.1492004, 434.8284302
5: -221.4252777, 172.0696564, -220.1589508, 171.1189880, -392.5442200, 392.2285767
6: -211.8826599, 203.2927246, -210.6185913, 202.0698242, -413.9524841, 413.9112854
7: -230.5145721, 193.7198944, -229.1683044, 192.5969543, -423.1115112, 422.8881836
8: -277.8388977, 189.5752106, -276.1866150, 188.4299011, -466.2687378, 465.7618408
9: -209.8368378, 206.5505676, -208.6018066, 205.3106689, -415.1475220, 415.1522827

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2213737, upper bound: 461.2211687
time: 11.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2209934, upper bound: 461.2209934
time: 8.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -227.1806793, 179.7659607, -225.1155090, 178.1533508, -405.3340454, 404.8814697
1: -189.8097687, 159.5156708, -188.0131226, 157.9938660, -347.8036194, 347.5288086
2: -249.4809570, 161.9442902, -247.1558075, 160.4508514, -409.9318237, 409.1000977
3: -265.3932800, 140.1294708, -262.9147339, 138.7195892, -404.1128540, 403.0441589
4: -243.3945160, 186.5373688, -241.1761169, 184.7895355, -428.1840515, 427.7135010
5: -218.2275238, 169.5514374, -216.2615051, 168.0119629, -386.2394714, 385.8129272
6: -208.8237457, 200.3593903, -206.8992767, 198.5030060, -407.3267517, 407.2586670
7: -227.1459503, 190.8996887, -225.0730286, 189.1523895, -416.2983093, 415.9727173
8: -273.8646851, 186.8535767, -271.2874451, 185.0843811, -458.9490662, 458.1410217
9: -206.7483673, 203.5579376, -204.8277740, 201.6409912, -408.3893433, 408.3857117

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2179270, upper bound: 461.2177625
time: 10.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2175827, upper bound: 461.2175935
time: 12.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -227.6767426, 180.1611938, -229.9343109, 181.9852142, -409.6619263, 410.0954590
1: -190.2305145, 159.8802032, -192.0685425, 161.4212341, -351.6517334, 351.9487305
2: -250.0333252, 162.3100586, -252.4712219, 163.8916473, -413.9249573, 414.7812500
3: -265.9970703, 140.4500122, -268.6133728, 141.7223206, -407.7193909, 409.0633850
4: -243.9278259, 186.9510956, -246.3487091, 188.7471924, -432.6750183, 433.2997742
5: -218.6991730, 169.9337616, -220.8875427, 171.6459656, -390.3451538, 390.8212891
6: -209.2910156, 200.8025665, -211.3398895, 202.7641602, -412.0551758, 412.1424561
7: -227.6615448, 191.3355560, -229.9342041, 193.2438812, -420.9053955, 421.2696533
8: -274.4692078, 187.2574463, -277.1035156, 189.0254211, -463.4945984, 464.3609009
9: -207.2283020, 204.0184174, -209.2802887, 205.9844513, -413.2127686, 413.2986145

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2184442, upper bound: 461.2185342
time: 11.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2180947, upper bound: 461.2183703
time: 11.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -230.0101776, 182.0015411, -226.7339325, 179.4323120, -409.4424744, 408.7354736
1: -192.1782837, 161.5088654, -189.3660431, 159.1321716, -351.3104553, 350.8748779
2: -252.5905914, 163.9678345, -248.9341431, 161.6028900, -414.1934509, 412.9019775
3: -268.7095032, 141.8653107, -264.8088989, 139.7077332, -408.4172363, 406.6741943
4: -246.4613647, 188.8697510, -242.9266815, 186.1210632, -432.5823975, 431.7964478
5: -220.9557953, 171.6885986, -217.8155823, 169.2313385, -390.1870422, 389.5041809
6: -211.4167938, 202.8510284, -208.3821716, 199.9295349, -411.3463135, 411.2332153
7: -230.0000305, 193.2866364, -226.7037506, 190.5126648, -420.5126953, 419.9903870
8: -277.2365417, 189.1719971, -273.2202759, 186.4102783, -463.6468201, 462.3922729
9: -209.3582916, 206.0918884, -206.3128510, 203.0901489, -412.4484253, 412.4047241

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2194880, upper bound: 461.2187051
time: 11.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2191098, upper bound: 461.2184930
time: 11.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -230.5041656, 182.3955078, -231.5914154, 183.2929382, -413.7971191, 413.9869385
1: -192.5975494, 161.8716888, -193.4535675, 162.5864258, -355.1839600, 355.3251953
2: -253.1396637, 164.3320007, -254.2921906, 165.0703278, -418.2099915, 418.6241760
3: -269.3107910, 142.1840515, -270.5528564, 142.7340393, -412.0447998, 412.7369080
4: -246.9949799, 189.2815247, -248.1411438, 190.1102448, -437.1052246, 437.4226685
5: -221.4252777, 172.0696564, -222.4782104, 172.8926392, -394.3179016, 394.5478210
6: -211.8826599, 203.2927246, -212.8588867, 204.2248535, -416.1075134, 416.1516113
7: -230.5145721, 193.7198944, -231.6026306, 194.6356659, -425.1502380, 425.3224792
8: -277.8388977, 189.5752106, -279.0832214, 190.3811035, -468.2200012, 468.6584473
9: -209.8368378, 206.5505676, -210.7991791, 207.4671631, -417.3040161, 417.3497314

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2199813, upper bound: 461.2195124
time: 10.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2196493, upper bound: 461.2193331
time: 10.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.59 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2188886, upper bound: 461.2189386
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2184925, upper bound: 461.2187722
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2194552, upper bound: 461.2198724
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2191131, upper bound: 461.2197084
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2209000, upper bound: 461.2203800
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2184925, upper bound: 461.2201866
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2213737, upper bound: 461.2211687
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2209934, upper bound: 461.2209934
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2179270, upper bound: 461.2177625
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2175827, upper bound: 461.2175935
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2184442, upper bound: 461.2185342
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2180947, upper bound: 461.2183703
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2194880, upper bound: 461.2187051
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2191098, upper bound: 461.2184930
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2199813, upper bound: 461.2195124
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -461.2196493, upper bound: 461.2193331

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -218.6217041, 172.9985657, -224.3908997, 177.5919189, -396.2136230, 397.3894653
1: -182.6924286, 153.5315704, -187.4895630, 157.5536804, -340.2460022, 341.0211182
2: -240.0743713, 155.9221039, -246.3859863, 159.9822235, -400.0565796, 402.3080750
3: -255.4605560, 134.9334869, -262.1501465, 138.3764343, -393.8369751, 397.0836182
4: -234.2128906, 179.4820709, -240.4235077, 184.2319489, -418.4448242, 419.9054871
5: -209.9895172, 163.1644745, -215.5857239, 167.5502167, -377.5397339, 378.7501526
6: -200.9468994, 192.8294067, -206.2416687, 197.8776855, -398.8244934, 399.0710754
7: -218.6142883, 183.7653351, -224.3967133, 188.6111145, -407.2254028, 408.1619873
8: -263.5375061, 179.7811432, -270.4630737, 184.5115204, -448.0490112, 450.2442017
9: -198.9882507, 195.9506836, -204.2555237, 201.0623169, -400.0505371, 400.2061768

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2157886, upper bound: 461.2165730
time: 10.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2154198, upper bound: 461.2158370
time: 11.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -223.0226593, 176.4496613, -221.6579895, 175.4341431, -398.4567261, 398.1076660
1: -186.3381500, 156.5548706, -185.2084045, 155.6331024, -341.9711914, 341.7632141
2: -244.8399200, 159.0261383, -243.3747711, 158.0624084, -402.9023438, 402.4008484
3: -260.5819092, 137.5632172, -258.9660034, 136.7014771, -397.2833862, 396.5291748
4: -238.9215851, 183.0306396, -237.4799194, 181.9780273, -420.8995667, 420.5105591
5: -214.1841888, 166.3978271, -212.9626312, 165.5170135, -379.7011719, 379.3604736
6: -204.9193268, 196.6962433, -203.7132721, 195.4677734, -400.3870850, 400.4094543
7: -222.9944916, 187.4418945, -221.6663513, 186.3375702, -409.3320007, 409.1082153
8: -268.7312927, 183.2471924, -267.1483765, 182.2459717, -450.9772644, 450.3955688
9: -202.9907227, 199.8464203, -201.7789917, 198.6267853, -401.6174927, 401.6254272

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2156176, upper bound: 461.2164610
time: 11.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2152010, upper bound: 461.2156727
time: 10.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -220.9784241, 174.8576660, -220.4803619, 174.4832458, -395.4616699, 395.3380127
1: -184.6607513, 155.1781311, -184.1850433, 154.7631989, -339.4239502, 339.3631592
2: -242.6579590, 157.5980530, -242.0602570, 157.1901093, -399.8480225, 399.6582947
3: -258.2015381, 136.3641663, -257.5050659, 135.9285431, -394.1300354, 393.8691406
4: -236.7713470, 181.4194794, -236.2367706, 181.0191956, -417.7905273, 417.6562500
5: -212.2685089, 164.9365082, -211.8345642, 164.6063538, -376.8748169, 376.7710571
6: -203.0951843, 194.8993988, -202.6266785, 194.4106445, -397.5057983, 397.5260620
7: -220.9785004, 185.7372437, -220.4505920, 185.2812805, -406.2597351, 406.1878357
8: -266.3337402, 181.7143097, -265.7182007, 181.3225708, -447.6563110, 447.4324951
9: -201.1389465, 198.0458374, -200.6367340, 197.5282745, -398.6671753, 398.6825562

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2154269, upper bound: 461.2174490
time: 13.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2171006, upper bound: 461.2167915
time: 11.25 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 25.95 seconds
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 7, lower bound: -461.2157886, upper bound: 461.2165730
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 7, lower bound: -461.2154198, upper bound: 461.2158370
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 7, lower bound: -461.2156176, upper bound: 461.2164610
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 7, lower bound: -461.2152010, upper bound: 461.2156727
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 7, lower bound: -461.2154269, upper bound: 461.2174490
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 7, lower bound: -461.2171006, upper bound: 461.2167915
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 7, lower bound: -461.2184925, upper bound: 461.2201866
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 7, lower bound: -461.2213737, upper bound: 461.2211687
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 7, lower bound: -461.2209934, upper bound: 461.2209934
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 7, lower bound: -461.2199813, upper bound: 461.2195124
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.95
Output dim: 7, lower bound: -461.2196493, upper bound: 461.2193331
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=464.3514404296875
rel_dist={7: [-461.23120259765585, 461.23120260223186]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2268732, upper bound: 461.2267972
time: 15.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2267187, upper bound: 461.2267187
time: 18.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 33.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 33.78
Output dim: 7, lower bound: -461.2268732, upper bound: 461.2267972
IS_A2, status: Status.UNKNOWN, split count: 1, time: 33.78
Output dim: 7, lower bound: -461.2267187, upper bound: 461.2267187

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -243.1105499, 192.3500366, -431.4244385, 432.2729492
1: -199.7922211, 167.9400635, -203.1938324, 170.7909698, -370.5831909, 371.1338501
2: -262.6058350, 170.4243469, -267.0626221, 173.2895813, -435.8954163, 437.4869385
3: -279.3811340, 147.4806366, -284.1234741, 149.9656372, -429.3467712, 431.6041260
4: -256.2637939, 196.3474579, -260.6112671, 199.6806946, -455.9444885, 456.9587402
5: -229.6463013, 178.5066833, -233.5380859, 181.5320129, -411.1783142, 412.0447388
6: -219.7590637, 210.8946838, -223.4725189, 214.4641418, -434.2232056, 434.3671875
7: -239.1515045, 200.9265594, -243.2075806, 204.3112335, -443.4627380, 444.1341553
8: -288.1658630, 196.6365509, -293.0315552, 199.9494324, -488.1152649, 489.6680908
9: -217.7042542, 214.2737274, -221.3847351, 217.8970642, -435.6013184, 435.6584473

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2248356, upper bound: 461.2246369
time: 17.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2256721, upper bound: 461.2255714
time: 18.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -242.0842743, 191.5462341, -238.9289856, 189.0464783, -431.1307373, 430.4751892
1: -202.2670288, 170.0345459, -199.6886749, 167.8608093, -370.1278076, 369.7232056
2: -265.9084778, 172.5523376, -262.4733887, 170.3368835, -436.2453613, 435.0256653
3: -282.9065552, 149.2875214, -279.2297974, 147.4079590, -430.3145142, 428.5173340
4: -259.4692688, 198.7713928, -256.1103821, 196.2464752, -455.7157593, 454.8817749
5: -232.5094452, 180.7077179, -229.5148621, 178.4202271, -410.9296875, 410.2225952
6: -222.5233765, 213.5542908, -219.6300964, 210.7787476, -433.3021240, 433.1843262
7: -242.1613312, 203.4472351, -239.0276031, 200.8197937, -442.9810791, 442.4748230
8: -291.7500916, 199.0496979, -287.9992981, 196.5101318, -488.2602234, 487.0490112
9: -220.4269104, 216.9405365, -217.5845642, 214.1495209, -434.5764160, 434.5250854

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2246952, upper bound: 461.2245676
time: 15.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2254990, upper bound: 461.2254990
time: 14.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.54
Output dim: 7, lower bound: -461.2248356, upper bound: 461.2246369
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.54
Output dim: 7, lower bound: -461.2256721, upper bound: 461.2255714
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.54
Output dim: 7, lower bound: -461.2246952, upper bound: 461.2245676
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.54
Output dim: 7, lower bound: -461.2254990, upper bound: 461.2254990

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -233.7174530, 184.9278870, -236.4645996, 187.0947571, -420.8121948, 421.3923950
1: -195.3067932, 164.1585083, -197.6219177, 166.0937347, -361.4005127, 361.7803650
2: -256.7077637, 166.6110382, -259.7414856, 168.5498047, -425.2575684, 426.3524780
3: -273.0987549, 144.2042236, -276.3261719, 145.8912048, -418.9899597, 420.5303955
4: -250.4503632, 191.9301605, -253.3928070, 194.1942291, -444.6445923, 445.3229675
5: -224.4990540, 174.4525757, -227.1426544, 176.4946747, -400.9937134, 401.5952148
6: -214.8432770, 206.1583557, -217.3731842, 208.5868378, -423.4301147, 423.5315247
7: -233.7356567, 196.4139252, -236.4824829, 198.7042999, -432.4399414, 432.8964233
8: -281.7436523, 192.2360687, -285.0640564, 194.4833832, -476.2270508, 477.3000793
9: -212.7705078, 209.4564209, -215.2522736, 211.9150848, -424.6856079, 424.7086792

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2201848, upper bound: 461.2200416
time: 17.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2201848, upper bound: 461.2205897
time: 15.34 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -236.3079071, 186.9738464, -239.2591095, 189.3032227, -425.6111450, 426.2329407
1: -197.4709625, 165.9806519, -199.9617615, 168.0627594, -365.5337219, 365.9422607
2: -259.5534058, 168.4569550, -262.8132324, 170.5504608, -430.1038208, 431.2701416
3: -276.1325378, 145.7857819, -279.6006775, 147.6057739, -423.7383118, 425.3864746
4: -253.2553558, 194.0615540, -256.4224548, 196.4981232, -449.7534790, 450.4840088
5: -226.9878693, 176.4043579, -229.8367004, 178.6052094, -405.5930786, 406.2409973
6: -217.2183990, 208.4413300, -219.9344177, 211.0482025, -428.2666016, 428.3757324
7: -236.3468323, 198.5932465, -239.3025360, 201.0626984, -437.4095459, 437.8957520
8: -284.8366089, 194.3558960, -288.3956604, 196.7738190, -481.6103821, 482.7515564
9: -215.1513519, 211.7755432, -217.8302765, 214.4181671, -429.5694580, 429.6058044

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2214186, upper bound: 461.2213912
time: 16.24 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2218872, upper bound: 461.2217807
time: 17.23 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -236.8509827, 187.4098206, -232.3316498, 183.8305817, -420.6815491, 419.7414551
1: -197.8856964, 166.3407440, -194.1573334, 163.1979065, -361.0836182, 360.4980469
2: -260.1481323, 168.8272400, -255.2072754, 165.6316833, -425.7797852, 424.0344849
3: -276.7715454, 146.0870209, -271.4927673, 143.3628845, -420.1344299, 417.5797729
4: -253.7940216, 194.4567871, -248.9468536, 190.8003387, -444.5943604, 443.4036255
5: -227.4828491, 176.7498016, -223.1668854, 173.4219971, -400.9048462, 399.9166870
6: -217.7216339, 208.9285736, -213.5756073, 204.9460449, -422.6676636, 422.5041809
7: -236.8727264, 199.0402679, -232.3542480, 195.2547913, -432.1275024, 431.3945312
8: -285.4780273, 194.7521362, -280.0932007, 191.0856018, -476.5636292, 474.8453369
9: -215.6117249, 212.2365112, -211.5001068, 208.2133789, -423.8251038, 423.7365417

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2200361, upper bound: 461.2199646
time: 14.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2206745, upper bound: 461.2205275
time: 16.12 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -239.2664795, 189.3177032, -234.9769287, 185.9217072, -425.1881714, 424.2946167
1: -199.9028320, 168.0386200, -196.3737183, 165.0618896, -364.9647217, 364.4122314
2: -262.8003845, 170.5475006, -258.1145020, 167.5254822, -430.3258667, 428.6619873
3: -279.5969543, 147.5606384, -274.5889282, 144.9868469, -424.5838013, 422.1495361
4: -256.4050598, 196.4423370, -251.8132935, 192.9809723, -449.3860168, 448.2554932
5: -229.8013306, 178.5654907, -225.7176208, 175.4168549, -405.2180786, 404.2831116
6: -219.9346619, 211.0553589, -216.0003357, 207.2744904, -427.2091675, 427.0556946
7: -239.3019714, 201.0699463, -235.0191040, 197.4860687, -436.7880249, 436.0890503
8: -288.3592834, 196.7266388, -283.2436218, 193.2524261, -481.6116943, 479.9702454
9: -217.8246002, 214.3952332, -213.9352112, 210.5799408, -428.4045410, 428.3304443

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2212531, upper bound: 461.2213215
time: 14.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2217060, upper bound: 461.2217060
time: 14.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 30.93 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2201848, upper bound: 461.2200416
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2201848, upper bound: 461.2205897
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2214186, upper bound: 461.2213912
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2218872, upper bound: 461.2217807
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2200361, upper bound: 461.2199646
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2206745, upper bound: 461.2205275
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2212531, upper bound: 461.2213215
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.93
Output dim: 7, lower bound: -461.2217060, upper bound: 461.2217060

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -219.3538513, 173.5952606, -225.7952271, 178.6762848, -398.0300903, 399.3905029
1: -183.2286835, 153.9562988, -188.6521606, 158.5158691, -341.7445679, 342.6083679
2: -240.8200073, 156.3688812, -247.9401855, 160.9417114, -401.7617188, 404.3089600
3: -256.1575012, 135.2325592, -263.7415466, 139.2280579, -395.3855591, 398.9741211
4: -234.9823303, 180.0935822, -241.9032745, 185.4033051, -420.3855286, 421.9968262
5: -210.7560577, 163.7303619, -216.9328918, 168.5317535, -379.2877808, 380.6632690
6: -201.6056366, 193.4072876, -207.5396423, 199.1162109, -400.7218628, 400.9469299
7: -219.2796326, 184.3146362, -225.7444611, 189.7175903, -408.9972229, 410.0590820
8: -264.4017944, 180.4146729, -272.1827087, 185.7032318, -450.1050110, 452.5973816
9: -199.5702362, 196.4953003, -205.4467468, 202.2889252, -401.8591614, 401.9420166

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2170525, upper bound: 461.2169912
time: 19.15 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2169940, upper bound: 461.2168552
time: 16.91 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -224.9810028, 178.0598602, -227.5457611, 180.0636749, -405.0446777, 405.6056213
1: -187.9687653, 157.9565277, -190.1306458, 159.7812195, -347.7499695, 348.0871582
2: -247.0337524, 160.3831482, -249.8815155, 162.2124023, -409.2461548, 410.2646484
3: -262.8144836, 138.7299042, -265.8403931, 140.3416901, -403.1561890, 404.5703125
4: -241.0265503, 184.7205658, -243.7869263, 186.8545837, -427.8811340, 428.5074463
5: -216.1560364, 167.9707794, -218.5990906, 169.8593903, -386.0154419, 386.5698853
6: -206.7952118, 198.3885651, -209.1718140, 200.6740875, -407.4692688, 407.5603638
7: -224.9579163, 189.0892334, -227.5352936, 191.2246704, -416.1825562, 416.6245117
8: -271.1975403, 185.0108948, -274.3067017, 187.1381836, -458.3357239, 459.3175354
9: -204.7678070, 201.5693359, -207.0998383, 203.8908997, -408.6586304, 408.6691895

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2178210, upper bound: 461.2176455
time: 15.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2177662, upper bound: 461.2175234
time: 14.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -221.9261475, 175.6275177, -228.6116486, 180.9030457, -402.8291931, 404.2391663
1: -185.3780823, 155.7652740, -191.0105896, 160.4998474, -345.8779297, 346.7758789
2: -243.6467743, 158.2005157, -251.0373230, 162.9565582, -406.6033325, 409.2378235
3: -259.1695862, 136.8030090, -267.0413513, 140.9561615, -400.1257324, 403.8443604
4: -237.7654724, 182.2105255, -244.9554443, 187.7251282, -425.4906006, 427.1659546
5: -213.2277374, 165.6685944, -219.6492767, 170.6579590, -383.8856812, 385.3178406
6: -203.9639282, 195.6739807, -210.1208344, 201.5970459, -405.5609436, 405.7947998
7: -221.8719482, 186.4784241, -228.5853729, 192.0935974, -413.9655457, 415.0637817
8: -267.4716797, 182.5190430, -275.5394592, 188.0111237, -455.4827271, 458.0584717
9: -201.9325256, 198.7978821, -208.0431519, 204.8109436, -406.7434387, 406.8410339

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2181002, upper bound: 461.2181854
time: 16.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2180392, upper bound: 461.2180388
time: 17.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -227.5782776, 180.1110687, -230.3416901, 182.2731476, -409.8514404, 410.4527588
1: -190.1389618, 159.7823944, -192.4715271, 161.7502136, -351.8891602, 352.2539062
2: -249.8866882, 162.2322845, -252.9537659, 164.2115021, -414.0981750, 415.1860352
3: -265.8547668, 140.3153076, -269.1153870, 142.0565033, -407.9112549, 409.4306946
4: -243.8393860, 186.8567810, -246.8195343, 189.1576233, -432.9969177, 433.6763306
5: -218.6507874, 169.9258881, -221.2934875, 171.9712067, -390.6220093, 391.2192688
6: -209.1760864, 200.6775513, -211.7325287, 203.1367188, -412.3127747, 412.4100647
7: -227.5763855, 191.2722626, -230.3562012, 193.5810547, -421.1574097, 421.6284790
8: -274.2973938, 187.1351776, -277.6388245, 189.4290771, -463.7264404, 464.7739868
9: -207.1527100, 203.8924866, -209.6772461, 206.3936157, -413.5463257, 413.5697021

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2189636, upper bound: 461.2189162
time: 12.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2189209, upper bound: 461.2188032
time: 16.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -222.7250519, 176.2637634, -221.8095856, 175.5296783, -398.2547302, 398.0733643
1: -186.0122070, 156.3072052, -185.3147278, 155.7254944, -341.7377014, 341.6219482
2: -244.5253754, 158.7496033, -243.5718689, 158.1264343, -402.6517944, 402.3214722
3: -260.1130066, 137.2577362, -259.0854492, 136.7895813, -396.9025879, 396.3432007
4: -238.5847015, 182.8191681, -237.6183624, 182.1335144, -420.7182007, 420.4374695
5: -213.9655914, 166.2045135, -213.0996094, 165.5691681, -379.5347595, 379.3040771
6: -204.7064972, 196.3905640, -203.8808289, 195.6075439, -400.3140259, 400.2713928
7: -222.6576843, 187.1398468, -221.7658386, 186.3918610, -409.0495300, 408.9057007
8: -268.4239807, 183.1210175, -267.3921204, 182.4241180, -450.8480835, 450.5131226
9: -202.6286011, 199.4930573, -201.8292542, 198.7221527, -401.3506470, 401.3223267

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2169541, upper bound: 461.2169813
time: 15.51 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2168732, upper bound: 461.2168479
time: 14.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -227.5391998, 180.0923767, -223.2736664, 176.6906738, -404.2298584, 403.3660278
1: -190.0633240, 159.7310944, -186.5467834, 156.7844543, -346.8477783, 346.2778015
2: -249.8351135, 162.1870880, -245.1930847, 159.1917114, -409.0267944, 407.3801880
3: -265.8059998, 140.2575531, -260.8395386, 137.7240143, -403.5300293, 401.0971069
4: -243.7515411, 186.7729187, -239.1923828, 183.3442535, -427.0957947, 425.9653015
5: -218.5875397, 169.8361206, -214.4882812, 166.6803284, -385.2678833, 384.3243408
6: -209.1422119, 200.6473846, -205.2445984, 196.9078369, -406.0500488, 405.8919678
7: -227.5141602, 191.2273712, -223.2655945, 187.6541443, -415.1683044, 414.4928894
8: -274.2336731, 187.0586243, -269.1651917, 183.6277313, -457.8613892, 456.2238159
9: -207.0776215, 203.8327484, -203.2167664, 200.0637360, -407.1413269, 407.0494995

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2176789, upper bound: 461.2176094
time: 16.80 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2176154, upper bound: 461.2174680
time: 16.68 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -225.1240845, 178.1592407, -224.4749603, 177.6361237, -402.7601929, 402.6342163
1: -188.0158234, 157.9917145, -187.5475159, 157.6021881, -345.6180115, 345.5391846
2: -247.1585846, 160.4575958, -246.5005188, 160.0334778, -407.1920776, 406.9580688
3: -262.9178467, 138.7211456, -262.2033081, 138.4246674, -401.3425293, 400.9244385
4: -241.1755219, 184.7905426, -240.5037231, 184.3296814, -425.5051880, 425.2942200
5: -216.2688446, 168.0075073, -215.6697388, 167.5781555, -383.8469849, 383.6772461
6: -206.9032898, 198.5018616, -206.3224640, 197.9526978, -404.8559875, 404.8243408
7: -225.0702667, 189.1546021, -224.4505463, 188.6383972, -413.7086792, 413.6051636
8: -271.2836609, 185.0833130, -270.5642700, 184.6075439, -455.8911438, 455.6475830
9: -204.8260193, 201.6359711, -204.2813873, 201.1051941, -405.9311829, 405.9173584

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2179336, upper bound: 461.2181225
time: 15.53 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2178465, upper bound: 461.2179677
time: 13.08 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -229.9614258, 182.0049744, -225.9278870, 178.7869873, -408.7483826, 407.9328613
1: -192.0869293, 161.4325867, -188.7697906, 158.6531372, -350.7400513, 350.2023926
2: -252.4944458, 163.9112549, -248.1078339, 161.0899353, -413.5843811, 412.0190125
3: -268.6384888, 141.7355957, -263.9450989, 139.3524323, -407.9909058, 405.6806946
4: -246.3691864, 188.7632294, -242.0653229, 185.5302582, -431.8993835, 430.8284912
5: -220.9126892, 171.6543427, -217.0465698, 168.6794739, -389.5921631, 388.7008057
6: -211.3613434, 202.7793579, -207.6758423, 199.2421722, -410.6034546, 410.4552002
7: -229.9488373, 193.2612305, -225.9374847, 189.8903046, -419.8391113, 419.1987000
8: -277.1227722, 189.0384216, -272.3242188, 185.8009033, -462.9236755, 461.3625793
9: -209.2939148, 205.9953003, -205.6570282, 202.4344177, -411.7283020, 411.6523132

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2187778, upper bound: 461.2188444
time: 16.05 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2187206, upper bound: 461.2187206
time: 16.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 33.87 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2170525, upper bound: 461.2169912
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2169940, upper bound: 461.2168552
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2178210, upper bound: 461.2176455
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2177662, upper bound: 461.2175234
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2181002, upper bound: 461.2181854
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2180392, upper bound: 461.2180388
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2189636, upper bound: 461.2189162
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2189209, upper bound: 461.2188032
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2169541, upper bound: 461.2169813
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2168732, upper bound: 461.2168479
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2176789, upper bound: 461.2176094
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2176154, upper bound: 461.2174680
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2179336, upper bound: 461.2181225
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2178465, upper bound: 461.2179677
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2187778, upper bound: 461.2188444
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 33.87
Output dim: 7, lower bound: -461.2187206, upper bound: 461.2187206
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=464.3514404296875
rel_dist={7: [-461.2309993444553, 461.2309993438606]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271315, upper bound: 461.2270101
time: 10.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2268618, upper bound: 461.2268618
time: 9.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.67
Output dim: 7, lower bound: -461.2271315, upper bound: 461.2270101
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.67
Output dim: 7, lower bound: -461.2268618, upper bound: 461.2268618

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -239.0743866, 189.1624146, -246.4925232, 195.0204315, -434.0948181, 435.6549072
1: -199.7922211, 167.9400635, -206.0443878, 173.1796112, -372.9718323, 373.9844360
2: -262.6058350, 170.4243469, -270.7961121, 175.6894531, -438.2952881, 441.2204285
3: -279.3811340, 147.4806366, -288.0963440, 152.0475922, -431.4287109, 435.5769653
4: -256.2637939, 196.3474579, -264.2546692, 202.4736023, -458.7373352, 460.6021118
5: -229.6463013, 178.5066833, -236.7992401, 184.0675812, -413.7138672, 415.3058472
6: -219.7590637, 210.8946838, -226.5833130, 217.4545898, -437.2136230, 437.4779663
7: -239.1515045, 200.9265594, -246.6055603, 207.1469879, -446.2984924, 447.5321045
8: -288.1658630, 196.6365509, -297.1071167, 202.7255402, -490.8914185, 493.7436523
9: -217.7042542, 214.2737274, -224.4688416, 220.9328613, -438.6371155, 438.7425537

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2253731, upper bound: 461.2249635
time: 11.81 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2260459, upper bound: 461.2258906
time: 10.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -242.0842743, 191.5462341, -242.2765198, 191.6910248, -433.7752686, 433.8226929
1: -202.2670288, 170.0345459, -202.5071259, 170.2206116, -372.4876404, 372.5416870
2: -265.9084778, 172.5523376, -266.1627808, 172.7102051, -438.6186523, 438.7150574
3: -282.9065552, 149.2875214, -283.1588745, 149.4659882, -432.3725586, 432.4464111
4: -259.4692688, 198.7713928, -259.7167358, 199.0050201, -458.4743042, 458.4881287
5: -232.5094452, 180.7077179, -232.7397919, 180.9231873, -413.4325867, 413.4475098
6: -222.5233765, 213.5542908, -222.7087097, 213.7366028, -436.2599792, 436.2629700
7: -242.1613312, 203.4472351, -242.3870697, 203.6232758, -445.7845154, 445.8342590
8: -291.7500916, 199.0496979, -292.0314636, 199.2592316, -491.0093079, 491.0811768
9: -220.4269104, 216.9405365, -220.6354523, 217.1534271, -437.5803223, 437.5759888

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2250816, upper bound: 461.2248328
time: 10.18 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2257391, upper bound: 461.2257391
time: 10.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -461.2253731, upper bound: 461.2249635
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -461.2260459, upper bound: 461.2258906
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -461.2250816, upper bound: 461.2248328
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.19
Output dim: 7, lower bound: -461.2257391, upper bound: 461.2257391

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -235.1092072, 186.0280151, -239.8443909, 189.7640228, -424.8732300, 425.8724060
1: -196.4721985, 165.1407166, -200.4700470, 168.4806213, -364.9528198, 365.6107483
2: -258.2399597, 167.6016388, -263.4725952, 170.9476929, -429.1876526, 431.0742188
3: -274.7308350, 145.0554199, -280.2969360, 147.9714813, -422.7023315, 425.3523560
4: -251.9605255, 193.0776520, -257.0340881, 196.9853363, -448.9458313, 450.1117554
5: -225.8361664, 175.5054016, -230.4018555, 179.0288849, -404.8650513, 405.9072571
6: -216.1205444, 207.3887329, -220.4818115, 211.5751495, -427.6956177, 427.8705444
7: -235.1424866, 197.5860443, -239.8785248, 201.5378113, -436.6802673, 437.4645691
8: -283.4118652, 193.3792877, -289.1362610, 197.2581482, -480.6700134, 482.5155334
9: -214.0516815, 210.7077026, -218.3338623, 214.9489594, -429.0006409, 429.0415344

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2208224, upper bound: 461.2205139
time: 9.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2213893, upper bound: 461.2209432
time: 13.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -237.1737976, 187.6588593, -242.6267242, 191.9633636, -429.1371460, 430.2855530
1: -198.1974945, 166.5939636, -202.8013611, 170.4418182, -368.6392822, 369.3952942
2: -260.5087891, 169.0727692, -266.5318909, 172.9406433, -433.4494324, 435.6046448
3: -277.1492004, 146.3162689, -283.5570374, 149.6794586, -426.8286438, 429.8732300
4: -254.1970367, 194.7770081, -260.0508423, 199.2797699, -453.4768066, 454.8278503
5: -227.8199463, 177.0623169, -233.0850372, 181.1304321, -408.9503784, 410.1473389
6: -218.0135498, 209.2092285, -223.0326233, 214.0265198, -432.0400696, 432.2418518
7: -237.2246704, 199.3235168, -242.6861877, 203.8869324, -441.1116028, 442.0096436
8: -285.8786926, 195.0697174, -292.4546814, 199.5384674, -485.4171143, 487.5244141
9: -215.9502869, 212.5574951, -220.9015961, 217.4415436, -433.3918457, 433.4590454

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2218540, upper bound: 461.2217894
time: 11.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2223160, upper bound: 461.2221083
time: 10.83 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -238.2110138, 188.4846802, -235.6763153, 186.4725342, -424.6835022, 424.1609497
1: -199.0244293, 167.3006287, -196.9728851, 165.5553131, -364.5797424, 364.2734680
2: -261.6449585, 169.7952271, -258.8937378, 168.0033264, -429.6482849, 428.6889648
3: -278.3657837, 146.9188080, -275.4175720, 145.4189453, -423.7847290, 422.3363647
4: -255.2687225, 195.5780792, -252.5495911, 193.5561371, -448.8247986, 448.1276855
5: -228.7889709, 177.7780914, -226.3886414, 175.9221039, -404.7110596, 404.1667480
6: -218.9696045, 210.1306458, -216.6510315, 207.9010620, -426.8706665, 426.7816772
7: -238.2470856, 200.1855164, -235.7104034, 198.0559845, -436.3030396, 435.8959351
8: -287.1077881, 195.8689270, -284.1209106, 193.8327789, -480.9405518, 479.9898376
9: -216.8628845, 213.4587097, -214.5477753, 211.2150116, -428.0778503, 428.0064697

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2205332, upper bound: 461.2203891
time: 10.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2210713, upper bound: 461.2208133
time: 9.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -240.1482849, 190.0150299, -238.3419342, 188.5802307, -428.7285156, 428.3568726
1: -200.6426544, 168.6633148, -199.2064056, 167.4342957, -368.0769653, 367.8697205
2: -263.7731018, 171.1748505, -261.8229370, 169.9116211, -433.6847229, 432.9978027
3: -280.6325989, 148.1011047, -278.5389404, 147.0556946, -427.6882935, 426.6400452
4: -257.3639526, 197.1712646, -255.4381104, 195.7541199, -453.1180725, 452.6093750
5: -230.6487274, 179.2359314, -228.9593506, 177.9331970, -408.5819092, 408.1952820
6: -220.7448425, 211.8373718, -219.0948792, 210.2476654, -430.9924927, 430.9322510
7: -240.1968689, 201.8139496, -238.3963623, 200.3043823, -440.5012512, 440.2103271
8: -289.4203796, 197.4536743, -287.2966919, 196.0159302, -485.4362793, 484.7503662
9: -218.6390228, 215.1918182, -217.0024109, 213.6001892, -432.2390747, 432.1942139

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2215252, upper bound: 461.2216588
time: 10.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2219683, upper bound: 461.2219683
time: 10.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2208224, upper bound: 461.2205139
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2213893, upper bound: 461.2209432
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2218540, upper bound: 461.2217894
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2223160, upper bound: 461.2221083
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2205332, upper bound: 461.2203891
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2210713, upper bound: 461.2208133
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2215252, upper bound: 461.2216588
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.28
Output dim: 7, lower bound: -461.2219683, upper bound: 461.2219683

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -220.7375183, 174.6889191, -232.1667328, 183.7054749, -404.4429932, 406.8556213
1: -184.3870087, 154.9327698, -194.0151215, 163.0271606, -347.4141541, 348.9478455
2: -242.3431244, 157.3535614, -254.9799957, 165.4731445, -407.8162231, 412.3335571
3: -257.7797241, 136.0786743, -271.2410889, 143.1768036, -400.9565430, 407.3197327
4: -236.4832611, 181.2344360, -248.7661285, 190.6597137, -427.1429749, 430.0005493
5: -212.0851288, 164.7771606, -223.0545197, 173.2986145, -385.3837280, 387.8316040
6: -202.8751526, 194.6304169, -213.4051056, 204.7598877, -407.6350403, 408.0354614
7: -220.6781464, 185.4799042, -232.1512909, 195.0709534, -415.7490540, 417.6311951
8: -266.0596924, 181.5510254, -279.8668823, 190.9397583, -456.9994507, 461.4179077
9: -200.8438873, 197.7391205, -211.2779236, 208.0223236, -408.8662109, 409.0169373

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2176052, upper bound: 461.2174613
time: 13.98 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2174920, upper bound: 461.2172147
time: 10.43 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -226.3697662, 179.1573639, -233.2267303, 184.5463867, -410.9161072, 412.3840332
1: -189.1315155, 158.9364319, -194.9114990, 163.7964325, -352.9279480, 353.8479004
2: -248.5624542, 161.3716736, -256.1563721, 166.2458954, -414.8082886, 417.5280457
3: -264.4426575, 139.5791321, -272.5149841, 143.8535614, -408.2962036, 412.0941162
4: -242.5335846, 185.8654022, -249.9070435, 191.5395050, -434.0730286, 435.7724304
5: -217.4898987, 169.0207977, -224.0620270, 174.1049500, -391.5948486, 393.0828247
6: -208.0696259, 199.6162720, -214.3966217, 205.7036896, -413.7733154, 414.0128784
7: -226.3618774, 190.2587738, -233.2400818, 195.9874725, -422.3493347, 423.4988403
8: -272.8616943, 186.1514435, -281.1547241, 191.8078461, -464.6695557, 467.3061523
9: -206.0460968, 202.8175812, -212.2842712, 208.9949188, -415.0410156, 415.1018372

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2183885, upper bound: 461.2180352
time: 12.96 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2182866, upper bound: 461.2178109
time: 12.24 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -222.7896271, 176.3101959, -234.9663086, 185.9190826, -408.7087097, 411.2764893
1: -186.1022491, 156.3767090, -196.3605347, 165.0005798, -351.1027832, 352.7372437
2: -244.5987854, 158.8146057, -258.0593567, 167.4766693, -412.0754395, 416.8739624
3: -260.1831360, 137.3317719, -274.5216675, 144.8951263, -405.0782471, 411.8534546
4: -238.7042999, 182.9238129, -251.8006592, 192.9679413, -431.6721802, 434.7244873
5: -214.0570068, 166.3244934, -225.7545166, 175.4123688, -389.4693604, 392.0789490
6: -204.7567139, 196.4394531, -215.9717712, 207.2266693, -411.9833984, 412.4111633
7: -222.7471313, 187.2065277, -234.9755859, 197.4334412, -420.1805420, 422.1821289
8: -268.5102234, 183.2307739, -283.2047119, 193.2339783, -461.7442017, 466.4354858
9: -202.7293854, 199.5774536, -213.8592682, 210.5296021, -413.2589722, 413.4367065

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2185464, upper bound: 461.2186978
time: 11.02 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2184352, upper bound: 461.2184249
time: 11.56 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -228.4374390, 180.7905273, -236.0126648, 186.7484283, -415.1858521, 416.8031921
1: -190.8596497, 160.3909912, -197.2452240, 165.7588501, -356.6184998, 357.6362305
2: -250.8343658, 162.8437347, -259.2189331, 168.2389221, -419.0733032, 422.0626831
3: -266.8635559, 140.8416138, -275.7782288, 145.5629120, -412.4264526, 416.6198425
4: -244.7736816, 187.5666351, -252.9285736, 193.8350830, -438.6087646, 440.4952087
5: -219.4759369, 170.5786591, -226.7479401, 176.2084045, -395.6843262, 397.3265991
6: -209.9653625, 201.4393158, -216.9491730, 208.1579437, -418.1232605, 418.3884888
7: -228.4474030, 191.9970551, -236.0505371, 198.3366394, -426.7840576, 428.0475464
8: -275.3310547, 187.8435822, -284.4764404, 194.0907593, -469.4217834, 472.3200073
9: -207.9455414, 204.6684418, -214.8528748, 211.4889526, -419.4345093, 419.5212708

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2194030, upper bound: 461.2193032
time: 11.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2193174, upper bound: 461.2190847
time: 117.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -224.0804901, 177.3351440, -228.0746918, 180.4740601, -404.5545044, 405.4098511
1: -187.1468048, 157.2635803, -190.5830383, 160.1564636, -347.3032532, 347.8466187
2: -246.0169067, 159.7141876, -250.4864655, 162.5815430, -408.5984497, 410.2006531
3: -261.7015991, 138.0866547, -266.4523621, 140.6700439, -402.3716431, 404.5390015
4: -240.0540314, 183.9363861, -244.3637238, 187.2942505, -427.3482666, 428.3001099
5: -215.2673950, 167.2292633, -219.1138153, 170.2476959, -385.5150452, 386.3430481
6: -205.9498444, 197.5883484, -209.6463165, 201.1537170, -407.1035767, 407.2346802
7: -224.0271759, 188.2810059, -228.0596008, 191.6524353, -415.6795959, 416.3406067
8: -270.0476685, 184.2342529, -274.9445496, 187.5763702, -457.6240234, 459.1787720
9: -203.8755341, 200.7109528, -207.5603790, 204.3570557, -408.2326050, 408.2713318

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2173518, upper bound: 461.2174004
time: 12.45 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2172046, upper bound: 461.2171527
time: 12.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -228.8965759, 181.1651154, -228.9775848, 181.1917114, -410.0882874, 410.1426697
1: -191.1997528, 160.6889496, -191.3456268, 160.8123016, -352.0120544, 352.0345764
2: -251.3291321, 163.1530457, -251.4885712, 163.2415619, -414.5706482, 414.6416016
3: -267.3970642, 141.0876312, -267.5384827, 141.2482452, -408.6453247, 408.6260986
4: -245.2234039, 187.8917847, -245.3361053, 188.0431976, -433.2666016, 433.2278748
5: -219.8909912, 170.8617706, -219.9710236, 170.9366302, -390.8276367, 390.8327942
6: -210.3877411, 201.8470612, -210.4904938, 201.9568329, -412.3445435, 412.3375549
7: -228.8857269, 192.3701477, -228.9895782, 192.4353790, -421.3210754, 421.3597107
8: -275.8601685, 188.1732788, -276.0394897, 188.3170013, -464.1771545, 464.2127075
9: -208.3259430, 205.0521393, -208.4215698, 205.1883698, -413.5142822, 413.4736938

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2180830, upper bound: 461.2179500
time: 11.08 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2179647, upper bound: 461.2176931
time: 11.31 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -226.0050201, 178.8558807, -230.7509766, 182.5905609, -408.5955811, 409.6068726
1: -188.7546692, 158.6157837, -192.8259125, 162.0425568, -350.7971802, 351.4416504
2: -248.1301270, 161.0843201, -253.4284058, 164.4966278, -412.6267700, 414.5127258
3: -263.9525452, 139.2609711, -269.5856934, 142.3132629, -406.2658081, 408.8466797
4: -242.1337280, 185.5186157, -247.2634125, 189.5007019, -431.6343994, 432.7820129
5: -217.1151733, 168.6771393, -221.6955872, 172.2671967, -389.3823853, 390.3727417
6: -207.7125549, 199.2830811, -212.0986176, 203.5100098, -411.2225647, 411.3816833
7: -225.9641266, 189.8976898, -230.7560425, 193.9094238, -419.8734436, 420.6537170
8: -272.3433838, 185.8094635, -278.1317139, 189.7687378, -462.1121216, 463.9411621
9: -205.6396332, 202.4317169, -210.0240631, 206.7513428, -412.3909912, 412.4557190

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2182230, upper bound: 461.2185772
time: 9.91 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2180619, upper bound: 461.2182926
time: 10.97 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -230.8657074, 182.7190399, -231.6592255, 183.3101959, -414.1759033, 414.3782654
1: -192.8448029, 162.0722961, -193.5915222, 162.7009735, -355.5457764, 355.6638184
2: -253.4916840, 164.5538788, -254.4332123, 165.1593323, -418.6510010, 418.9870911
3: -269.7003784, 142.2893219, -270.6770325, 142.8931274, -412.5935059, 412.9663696
4: -247.3518219, 189.5103912, -248.2407227, 190.2525024, -437.6043091, 437.7510986
5: -221.7809753, 172.3409119, -222.5553131, 172.9580536, -394.7390137, 394.8962402
6: -212.1921234, 203.5812073, -212.9472198, 204.3156281, -416.5077515, 416.5284119
7: -230.8660736, 194.0232391, -231.6900177, 194.6946106, -425.5606689, 425.7132568
8: -278.2102356, 189.7828064, -279.2327271, 190.5135345, -468.7237549, 469.0155334
9: -210.1287384, 206.8115845, -210.8885193, 207.5852509, -417.7139587, 417.7000732

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 75

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2190438, upper bound: 461.2191681
time: 10.38 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2189226, upper bound: 461.2189226
time: 10.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.28 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2176052, upper bound: 461.2174613
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2174920, upper bound: 461.2172147
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2183885, upper bound: 461.2180352
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2182866, upper bound: 461.2178109
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2185464, upper bound: 461.2186978
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2184352, upper bound: 461.2184249
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2194030, upper bound: 461.2193032
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2193174, upper bound: 461.2190847
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2173518, upper bound: 461.2174004
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2172046, upper bound: 461.2171527
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2180830, upper bound: 461.2179500
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2179647, upper bound: 461.2176931
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2182230, upper bound: 461.2185772
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2180619, upper bound: 461.2182926
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2190438, upper bound: 461.2191681
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 7, lower bound: -461.2189226, upper bound: 461.2189226
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=464.3514404296875
rel_dist={7: [-461.23112015200337, 461.23112008001283]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2200.90 seconds
