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
execution time: IAR + LP analysis = 1.14 + 11.64 = 12.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -461.2313825, upper bound: 461.2313824


# Binary Search by BASE starts (time budget: 2687.22 seconds, max iter: 100)

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
Binary search time: 50.09 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2637.13 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2121761, upper bound: 461.2158202
time: 9.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2313237, upper bound: 461.2313237
time: 8.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.27 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 18.27
Output dim: 7, lower bound: -461.2121761, upper bound: 461.2158202
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.27
Output dim: 7, lower bound: -461.2313237, upper bound: 461.2313237

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -252.2428284, 199.5615692, -451.7855835, 451.7895203
1: -210.8763428, 177.2277527, -210.8920135, 177.2409058, -388.1171875, 388.1196899
2: -277.1233826, 179.7556458, -277.1440125, 179.7689209, -456.8922729, 456.8995667
3: -294.8307190, 155.5760193, -294.8527222, 155.5875702, -450.4182739, 450.4287109
4: -270.4299011, 207.2068634, -270.4500427, 207.2221527, -477.6520386, 477.6569214
5: -242.3272858, 188.3644562, -242.3452606, 188.3783875, -430.7056580, 430.7096863
6: -231.8549500, 222.5227814, -231.8721466, 222.5393829, -454.3942871, 454.3949280
7: -252.3645477, 211.9523926, -252.3833771, 211.9680939, -464.3326111, 464.3357544
8: -304.0132751, 207.4294128, -304.0357361, 207.4447021, -511.4579773, 511.4651489
9: -229.6954498, 226.0778656, -229.7125092, 226.0945892, -455.7900391, 455.7903442

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2298255, upper bound: 461.2291509
time: 9.93 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2313237, upper bound: 461.2313237
time: 8.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.42 seconds
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.42
Output dim: 7, lower bound: -461.2298255, upper bound: 461.2291509
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.42
Output dim: 7, lower bound: -461.2313237, upper bound: 461.2313237

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -250.9974213, 198.5712433, -248.3846436, 196.4823303, -447.4797363, 446.9558716
1: -209.8439331, 176.3581238, -207.6110077, 174.4547577, -384.2986450, 383.9690552
2: -275.7655334, 178.8772125, -272.8272400, 176.9596252, -452.7250977, 451.7044373
3: -293.3777466, 154.8156281, -290.2131042, 153.1535034, -446.5312500, 445.0287476
4: -269.0953064, 206.1859741, -266.1643677, 203.9135590, -473.0088501, 472.3503418
5: -241.1427612, 187.4280701, -238.5716400, 185.2927704, -426.4355469, 425.9996643
6: -230.7187805, 221.4317169, -228.2691040, 219.0670929, -449.7858582, 449.7008057
7: -251.1222382, 210.9075470, -248.3985901, 208.5897369, -459.7119446, 459.3060608
8: -302.5299988, 206.4189148, -299.3418579, 204.2326660, -506.7626648, 505.7607422
9: -228.5522614, 224.9596252, -225.9736633, 222.4878540, -451.0401001, 450.9332886

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2238404, upper bound: 461.2227439
time: 9.24 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2233446, upper bound: 461.2224146
time: 11.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -250.1918945, 197.9295502, -450.1536255, 449.7385559
1: -210.8763428, 177.2277527, -209.1636200, 175.7812805, -386.6576233, 386.3913574
2: -277.1233826, 179.7556458, -274.8676453, 178.2964783, -455.4198608, 454.6231995
3: -294.8307190, 155.5760193, -292.4182129, 154.3156586, -449.1463623, 447.9942017
4: -270.4299011, 207.2068634, -268.2095947, 205.5055847, -475.9354858, 475.4164429
5: -242.3272858, 188.3644562, -240.3618317, 186.8012238, -429.1285095, 428.7262878
6: -231.8549500, 222.5227814, -229.9688568, 220.7111969, -452.5660706, 452.4916382
7: -252.3645477, 211.9523926, -250.3006287, 210.2151794, -462.5797119, 462.2530212
8: -304.0132751, 207.4294128, -301.5541687, 205.7552185, -509.7684631, 508.9835510
9: -229.6954498, 226.0778656, -227.7944031, 224.2203979, -453.9157715, 453.8722534

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2190425, upper bound: 461.2198253
time: 11.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2313048, upper bound: 461.2313048
time: 9.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.30 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 7, lower bound: -461.2238404, upper bound: 461.2227439
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 7, lower bound: -461.2233446, upper bound: 461.2224146
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 7, lower bound: -461.2190425, upper bound: 461.2198253
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 7, lower bound: -461.2313048, upper bound: 461.2313048

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -237.8613739, 188.1977997, -248.3846436, 196.4823303, -434.3436584, 436.5824585
1: -198.7715149, 167.0803680, -207.6110077, 174.4547577, -373.2262573, 374.6913147
2: -261.2634277, 169.5558777, -272.8272400, 176.9596252, -438.2229614, 442.3831177
3: -277.9450989, 146.7292023, -290.2131042, 153.1535034, -431.0986023, 436.9423218
4: -254.9441681, 195.3383484, -266.1643677, 203.9135590, -458.8577271, 461.5027161
5: -228.4750824, 177.5810394, -238.5716400, 185.2927704, -413.7678528, 416.1526489
6: -218.6358643, 209.8159332, -228.2691040, 219.0670929, -437.7029114, 438.0850220
7: -237.9239502, 199.8936615, -248.3985901, 208.5897369, -446.5136719, 448.2921753
8: -286.6995544, 195.6374664, -299.3418579, 204.2326660, -490.9322205, 494.9792786
9: -216.5747833, 213.1683960, -225.9736633, 222.4878540, -439.0626221, 439.1420593

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2022004, upper bound: 461.2014094
time: 10.78 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2238404, upper bound: 461.2227419
time: 10.06 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -240.8452606, 190.5606537, -245.7937927, 194.4358978, -435.2811584, 436.3544312
1: -201.2239685, 169.1559448, -205.4292755, 172.6276245, -373.8515625, 374.5852051
2: -264.5372009, 171.6653595, -269.9710693, 175.1231232, -439.6602783, 441.6363831
3: -281.4399414, 148.5201263, -287.1719055, 151.5613251, -433.0012817, 435.6920166
4: -258.1207581, 197.7405548, -263.3724976, 201.7751617, -459.8959351, 461.1130371
5: -231.3129120, 179.7618561, -236.0733185, 183.3512115, -414.6641235, 415.8351135
6: -221.3760986, 212.4520569, -225.8863220, 216.7774506, -438.1535339, 438.3383484
7: -240.9075623, 202.3924408, -245.7983246, 206.4188995, -447.3264160, 448.1907043
8: -290.2521057, 198.0287933, -296.2200623, 202.1051025, -492.3572083, 494.2488403
9: -219.2735901, 215.8114777, -223.6123352, 220.1632538, -439.4368286, 439.4238281

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1940358, upper bound: 461.1930863
time: 11.60 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2233439, upper bound: 461.2224122
time: 9.74 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -245.8076935, 194.5331116, -248.5894775, 196.6682739, -442.4759216, 443.1225281
1: -205.4902954, 172.7828369, -207.8220825, 174.6657104, -380.1560059, 380.6049194
2: -270.1727295, 175.1927948, -273.1153870, 177.1640015, -447.3367004, 448.3081665
3: -287.6137085, 151.6173859, -290.5716858, 153.3346558, -440.9483337, 442.1890869
4: -263.7218628, 202.0310516, -266.5074463, 204.2016907, -467.9235229, 468.5385132
5: -236.2254181, 183.7203217, -238.8281250, 185.6240540, -421.8494873, 422.5484314
6: -225.9941101, 216.9605408, -228.5020294, 219.3061371, -445.3002319, 445.4625854
7: -246.0734406, 206.7366333, -248.7107086, 208.8894958, -454.9629211, 455.4472961
8: -296.2528687, 202.1023102, -299.6235352, 204.4405823, -500.6934509, 501.7258301
9: -224.0952759, 220.3872681, -226.3618317, 222.7964783, -446.8917236, 446.7490845

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2074578, upper bound: 461.2084465
time: 12.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2071618, upper bound: 461.2079602
time: 10.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -250.6357727, 198.2948914, -250.1918945, 197.9295502, -448.5653076, 448.4867249
1: -209.5454254, 176.1207581, -209.1636200, 175.7812805, -385.3266907, 385.2843628
2: -275.3852844, 178.6346741, -274.8676453, 178.2964783, -453.6817627, 453.5023193
3: -293.0002747, 154.6041870, -292.4182129, 154.3156586, -447.3159180, 447.0223999
4: -268.7393188, 205.9131622, -268.2095947, 205.5055847, -474.2449036, 474.1227417
5: -240.8068542, 187.1950073, -240.3618317, 186.8012238, -427.6080933, 427.5568237
6: -230.4007263, 221.1276703, -229.9688568, 220.7111969, -451.1118774, 451.0965271
7: -250.7872467, 210.6373749, -250.3006287, 210.2151794, -461.0024414, 460.9379883
8: -302.0978699, 206.1241455, -301.5541687, 205.7552185, -507.8529968, 507.6783142
9: -228.2722473, 224.6627960, -227.7944031, 224.2203979, -452.4926453, 452.4571838

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2276167, upper bound: 461.2274133
time: 9.50 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270512, upper bound: 461.2270512
time: 8.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.44 seconds
IS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.2022004, upper bound: 461.2014094
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.2238404, upper bound: 461.2227419
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.1940358, upper bound: 461.1930863
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.2233439, upper bound: 461.2224122
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.2074578, upper bound: 461.2084465
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.2071618, upper bound: 461.2079602
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.2276167, upper bound: 461.2274133
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 19.44
Output dim: 7, lower bound: -461.2270512, upper bound: 461.2270512

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -236.2623291, 186.9372559, -248.3846436, 196.4823303, -432.7445679, 435.3218994
1: -197.4310760, 165.9657898, -207.6110077, 174.4547577, -371.8858032, 373.5767517
2: -259.5134277, 168.4277344, -272.8272400, 176.9596252, -436.4729919, 441.2549438
3: -276.1019897, 145.7508392, -290.2131042, 153.1535034, -429.2554932, 435.9639282
4: -253.2419128, 194.0349884, -266.1643677, 203.9135590, -457.1554565, 460.1993408
5: -226.9440308, 176.4031677, -238.5716400, 185.2927704, -412.2367859, 414.9747925
6: -217.1719666, 208.4112854, -228.2691040, 219.0670929, -436.2390747, 436.6803894
7: -236.3356781, 198.5694580, -248.3985901, 208.5897369, -444.9254150, 446.9679260
8: -284.7715454, 194.3229218, -299.3418579, 204.2326660, -489.0042114, 493.6647644
9: -215.1416626, 211.7430573, -225.9736633, 222.4878540, -437.6294861, 437.7167358

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2039712, upper bound: 461.2054430
time: 11.79 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1855294, upper bound: 461.1838608
time: 8.41 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -239.2938232, 189.3381348, -245.7937927, 194.4358978, -433.7297363, 435.1319275
1: -199.9243164, 168.0747528, -205.4292755, 172.6276245, -372.5519104, 373.5040283
2: -262.8397827, 170.5704651, -269.9710693, 175.1231232, -437.9628601, 440.5415039
3: -279.6516418, 147.5697784, -287.1719055, 151.5613251, -431.2129517, 434.7416992
4: -256.4701233, 196.4771729, -263.3724976, 201.7751617, -458.2453003, 459.8496399
5: -229.8280792, 178.6200256, -236.0733185, 183.3512115, -413.1792603, 414.6933289
6: -219.9554749, 211.0894623, -225.8863220, 216.7774506, -436.7329102, 436.9757385
7: -239.3672333, 201.1077881, -245.7983246, 206.4188995, -445.7861023, 446.9060974
8: -288.3806763, 196.7541504, -296.2200623, 202.1051025, -490.4857483, 492.9741821
9: -217.8839569, 214.4293823, -223.6123352, 220.1632538, -438.0471497, 438.0417175

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2057801, upper bound: 461.2067010
time: 11.55 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1863493, upper bound: 461.1845194
time: 9.65 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -237.4539185, 187.8849792, -250.1918945, 197.9295502, -435.3834839, 438.0767822
1: -198.4339905, 166.8104706, -209.1636200, 175.7812805, -374.2152710, 375.9740906
2: -260.8323059, 169.2810059, -274.8676453, 178.2964783, -439.1287842, 444.1486206
3: -277.5128174, 146.4890900, -292.4182129, 154.3156586, -431.8284912, 438.9072876
4: -254.5385437, 195.0266724, -268.2095947, 205.5055847, -460.0440979, 463.2362671
5: -228.0948029, 177.3129730, -240.3618317, 186.8012238, -414.8960266, 417.6748047
6: -218.2754211, 209.4711304, -229.9688568, 220.7111969, -438.9866333, 439.4400024
7: -237.5415649, 199.5843964, -250.3006287, 210.2151794, -447.7567444, 449.8850098
8: -286.2122803, 195.3045654, -301.5541687, 205.7552185, -491.9674683, 496.8587341
9: -216.2514801, 212.8292694, -227.7944031, 224.2203979, -440.4718628, 440.6236572

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2108188, upper bound: 461.2135523
time: 10.37 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2068354, upper bound: 461.2073479
time: 9.40 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -240.5147095, 190.3094177, -247.5883789, 195.8740540, -436.3887634, 437.8977966
1: -200.9523315, 168.9407043, -206.9720459, 173.9473267, -374.8996277, 375.9127502
2: -264.1912231, 171.4446259, -271.9984131, 176.4522400, -440.6434326, 443.4430542
3: -281.0971375, 148.3260345, -289.3632202, 152.7160339, -433.8131409, 437.6892700
4: -257.7991943, 197.4932709, -265.4057007, 203.3584137, -461.1575928, 462.8989868
5: -231.0072327, 179.5525208, -237.8525238, 184.8528442, -415.8600769, 417.4049683
6: -221.0862427, 212.1757050, -227.5751801, 218.4113007, -439.4974976, 439.7508850
7: -240.6028595, 202.1474915, -247.6888275, 208.0349731, -448.6378174, 449.8362427
8: -289.8569946, 197.7603302, -298.4178772, 203.6177979, -493.4747925, 496.1782227
9: -219.0208435, 215.5422974, -225.4234009, 221.8849182, -440.9057617, 440.9656982

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2124752, upper bound: 461.2147472
time: 8.92 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2086301, upper bound: 461.2086301
time: 8.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.27 seconds
IS_A2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.2039712, upper bound: 461.2054430
IS_A2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.1855294, upper bound: 461.1838608
IS_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.2057801, upper bound: 461.2067010
IS_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.1863493, upper bound: 461.1845194
IS_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.2108188, upper bound: 461.2135523
IS_A2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.2068354, upper bound: 461.2073479
IS_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.2124752, upper bound: 461.2147472
IS_A2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 7, lower bound: -461.2086301, upper bound: 461.2086301
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=464.3514404296875
rel_dist={7: [-461.2313236619183, 461.23132366191817]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2138482, upper bound: 461.2186490
time: 8.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2313564, upper bound: 461.2313561
time: 8.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.27 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 17.27
Output dim: 7, lower bound: -461.2138482, upper bound: 461.2186490
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.27
Output dim: 7, lower bound: -461.2313564, upper bound: 461.2313561

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -252.2428284, 199.5615692, -451.7855835, 451.7895203
1: -210.8763428, 177.2277527, -210.8920135, 177.2409058, -388.1171875, 388.1196899
2: -277.1233826, 179.7556458, -277.1440125, 179.7689209, -456.8922729, 456.8995667
3: -294.8307190, 155.5760193, -294.8527222, 155.5875702, -450.4182739, 450.4287109
4: -270.4299011, 207.2068634, -270.4500427, 207.2221527, -477.6520386, 477.6569214
5: -242.3272858, 188.3644562, -242.3452606, 188.3783875, -430.7056580, 430.7096863
6: -231.8549500, 222.5227814, -231.8721466, 222.5393829, -454.3942871, 454.3949280
7: -252.3645477, 211.9523926, -252.3833771, 211.9680939, -464.3326111, 464.3357544
8: -304.0132751, 207.4294128, -304.0357361, 207.4447021, -511.4579773, 511.4651489
9: -229.6954498, 226.0778656, -229.7125092, 226.0945892, -455.7900391, 455.7903442

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2301821, upper bound: 461.2293778
time: 10.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2313561, upper bound: 461.2313561
time: 8.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.79 seconds
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.79
Output dim: 7, lower bound: -461.2301821, upper bound: 461.2293778
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.79
Output dim: 7, lower bound: -461.2313561, upper bound: 461.2313561

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -251.9619751, 199.3383026, -248.3846436, 196.4823303, -448.4442749, 447.7229614
1: -210.6557007, 177.0419159, -207.6110077, 174.4547577, -385.1104736, 384.6528320
2: -276.8332520, 179.5679474, -272.8272400, 176.9596252, -453.7928162, 452.3951721
3: -294.5201721, 155.4135132, -290.2131042, 153.1535034, -447.6736755, 445.6266174
4: -270.1447144, 206.9886475, -266.1643677, 203.9135590, -474.0582886, 473.1530151
5: -242.0741272, 188.1643372, -238.5716400, 185.2927704, -427.3668823, 426.7359619
6: -231.6121368, 222.2896118, -228.2691040, 219.0670929, -450.6791992, 450.5586853
7: -252.0990601, 211.7290955, -248.3985901, 208.5897369, -460.6887817, 460.1276245
8: -303.6963501, 207.2134705, -299.3418579, 204.2326660, -507.9290161, 506.5553284
9: -229.4511108, 225.8388824, -225.9736633, 222.4878540, -451.9389343, 451.8125305

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2248177, upper bound: 461.2234304
time: 9.21 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2241394, upper bound: 461.2229675
time: 10.23 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -250.1918945, 197.9295502, -450.1536255, 449.7385559
1: -210.8763428, 177.2277527, -209.1636200, 175.7812805, -386.6576233, 386.3913574
2: -277.1233826, 179.7556458, -274.8676453, 178.2964783, -455.4198608, 454.6231995
3: -294.8307190, 155.5760193, -292.4182129, 154.3156586, -449.1463623, 447.9942017
4: -270.4299011, 207.2068634, -268.2095947, 205.5055847, -475.9354858, 475.4164429
5: -242.3272858, 188.3644562, -240.3618317, 186.8012238, -429.1285095, 428.7262878
6: -231.8549500, 222.5227814, -229.9688568, 220.7111969, -452.5660706, 452.4916382
7: -252.3645477, 211.9523926, -250.3006287, 210.2151794, -462.5797119, 462.2530212
8: -304.0132751, 207.4294128, -301.5541687, 205.7552185, -509.7684631, 508.9835510
9: -229.6954498, 226.0778656, -227.7944031, 224.2203979, -453.9157715, 453.8722534

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2278268, upper bound: 461.2275974
time: 10.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271223, upper bound: 461.2271223
time: 9.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.85 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.85
Output dim: 7, lower bound: -461.2248177, upper bound: 461.2234304
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.85
Output dim: 7, lower bound: -461.2241394, upper bound: 461.2229675
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.85
Output dim: 7, lower bound: -461.2278268, upper bound: 461.2275974
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.85
Output dim: 7, lower bound: -461.2271223, upper bound: 461.2271223

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -238.7978973, 188.9425812, -248.3846436, 196.4823303, -435.2802124, 437.3272095
1: -199.5597229, 167.7442169, -207.6110077, 174.4547577, -374.0144653, 375.3551941
2: -262.2998657, 170.2265015, -272.8272400, 176.9596252, -439.2594299, 443.0537109
3: -279.0538025, 147.3093719, -290.2131042, 153.1535034, -432.2073059, 437.5224609
4: -255.9631805, 196.1176758, -266.1643677, 203.9135590, -459.8767395, 462.2819824
5: -229.3794250, 178.2959595, -238.5716400, 185.2927704, -414.6721802, 416.8676147
6: -219.5031433, 210.6488647, -228.2691040, 219.0670929, -438.5702209, 438.9179688
7: -238.8717194, 200.6913300, -248.3985901, 208.5897369, -447.4614563, 449.0898438
8: -287.8318176, 196.4089661, -299.3418579, 204.2326660, -492.0644836, 495.7508240
9: -217.4470673, 214.0220032, -225.9736633, 222.4878540, -439.9348755, 439.9956665

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2115233, upper bound: 461.2108193
time: 10.52 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2248177, upper bound: 461.2234250
time: 9.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -241.8033600, 191.3228912, -248.3846436, 196.4823303, -438.2856140, 439.7075195
1: -202.0306396, 169.8354950, -207.6110077, 174.4547577, -376.4854126, 377.4464417
2: -265.5976868, 172.3513641, -272.8272400, 176.9596252, -442.5572510, 445.1785889
3: -282.5741577, 149.1136017, -290.2131042, 153.1535034, -435.7276611, 439.3267212
4: -259.1638184, 198.5378876, -266.1643677, 203.9135590, -463.0773315, 464.7022400
5: -232.2382812, 180.4936218, -238.5716400, 185.2927704, -417.5310364, 419.0652466
6: -222.2633667, 213.3044434, -228.2691040, 219.0670929, -441.3304443, 441.5735474
7: -241.8771973, 203.2083435, -248.3985901, 208.5897369, -450.4669189, 451.6068420
8: -291.4105530, 198.8183746, -299.3418579, 204.2326660, -495.6432190, 498.1602173
9: -220.1657257, 216.6848297, -225.9736633, 222.4878540, -442.6535645, 442.6585083

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2041267, upper bound: 461.2039446
time: 9.34 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2241394, upper bound: 461.2229638
time: 9.73 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -239.0557861, 189.1477356, -250.1918945, 197.9295502, -436.9853516, 439.3395081
1: -199.7766724, 167.9270020, -209.1636200, 175.7812805, -375.5579529, 377.0906372
2: -262.5853577, 170.4111481, -274.8676453, 178.2964783, -440.8818359, 445.2787170
3: -279.3592834, 147.4691925, -292.4182129, 154.3156586, -433.6749268, 439.8873901
4: -256.2437134, 196.3322906, -268.2095947, 205.5055847, -461.7492981, 464.5418701
5: -229.6284485, 178.4928589, -240.3618317, 186.8012238, -416.4296875, 418.8546448
6: -219.7420044, 210.8782043, -229.9688568, 220.7111969, -440.4531250, 440.8470459
7: -239.1328278, 200.9109955, -250.3006287, 210.2151794, -449.3480225, 451.2116089
8: -288.1436157, 196.6213989, -301.5541687, 205.7552185, -493.8988037, 498.1755676
9: -217.6873322, 214.2570953, -227.7944031, 224.2203979, -441.9077148, 442.0514832

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2114868, upper bound: 461.2100054
time: 8.82 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2278125, upper bound: 461.2275877
time: 10.28 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -242.0653381, 191.5312653, -250.1918945, 197.9295502, -439.9948730, 441.7230835
1: -202.2512054, 170.0212555, -209.1636200, 175.7812805, -378.0324707, 379.1848755
2: -265.8876343, 172.5388794, -274.8676453, 178.2964783, -444.1841125, 447.4064941
3: -282.8842773, 149.2758636, -292.4182129, 154.3156586, -437.1999512, 441.6940918
4: -259.4489136, 198.7559357, -268.2095947, 205.5055847, -464.9544983, 466.9655151
5: -232.4912720, 180.6936493, -240.3618317, 186.8012238, -419.2924805, 421.0554810
6: -222.5059814, 213.5375061, -229.9688568, 220.7111969, -443.2171631, 443.5063477
7: -242.1423340, 203.4313660, -250.3006287, 210.2151794, -452.3574829, 453.7319946
8: -291.7273254, 199.0342255, -301.5541687, 205.7552185, -497.4825134, 500.5883484
9: -220.4096527, 216.9235992, -227.7944031, 224.2203979, -444.6300354, 444.7180176

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2108482, upper bound: 461.2095549
time: 11.29 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271075, upper bound: 461.2271075
time: 8.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.84 seconds
IS_A2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2115233, upper bound: 461.2108193
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2248177, upper bound: 461.2234250
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2041267, upper bound: 461.2039446
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2241394, upper bound: 461.2229638
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2114868, upper bound: 461.2100054
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2278125, upper bound: 461.2275877
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2108482, upper bound: 461.2095549
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.84
Output dim: 7, lower bound: -461.2271075, upper bound: 461.2271075

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -237.1957703, 187.6796570, -248.3846436, 196.4823303, -433.6780701, 436.0643005
1: -198.2167969, 166.6274414, -207.6110077, 174.4547577, -372.6715393, 374.2383728
2: -260.5465088, 169.0961456, -272.8272400, 176.9596252, -437.5060730, 441.9234009
3: -277.2070312, 146.3290863, -290.2131042, 153.1535034, -430.3605347, 436.5421753
4: -254.2576904, 194.8118591, -266.1643677, 203.9135590, -458.1712646, 460.9762268
5: -227.8454895, 177.1158905, -238.5716400, 185.2927704, -413.1382446, 415.6875305
6: -218.0363464, 209.2415771, -228.2691040, 219.0670929, -437.1034241, 437.5106812
7: -237.2802124, 199.3645020, -248.3985901, 208.5897369, -445.8699341, 447.7630310
8: -285.9001465, 195.0919037, -299.3418579, 204.2326660, -490.1328125, 494.4337158
9: -216.0109711, 212.5939484, -225.9736633, 222.4878540, -438.4988403, 438.5675659

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2224367, upper bound: 461.2213545
time: 11.45 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2205328, upper bound: 461.2184812
time: 10.42 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -240.2525635, 190.1009216, -248.3846436, 196.4823303, -436.7348938, 438.4855652
1: -200.7316284, 168.7547913, -207.6110077, 174.4547577, -375.1863403, 376.3657532
2: -263.9010315, 171.2569733, -272.8272400, 176.9596252, -440.8605957, 444.0841675
3: -280.7868042, 148.1636963, -290.2131042, 153.1535034, -433.9403076, 438.3768005
4: -257.5139160, 197.2751007, -266.1643677, 203.9135590, -461.4274902, 463.4394531
5: -230.7540436, 179.3523102, -238.5716400, 185.2927704, -416.0467834, 417.9239502
6: -220.8434448, 211.9424744, -228.2691040, 219.0670929, -439.9105225, 440.2115784
7: -240.3375397, 201.9243011, -248.3985901, 208.5897369, -448.9272766, 450.3228455
8: -289.5400085, 197.5443268, -299.3418579, 204.2326660, -493.7726746, 496.8861694
9: -218.7767639, 215.3033447, -225.9736633, 222.4878540, -441.2645874, 441.2770081

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2220055, upper bound: 461.2197122
time: 11.00 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2225099, upper bound: 461.2209312
time: 9.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -239.0557861, 189.1477356, -248.6029663, 196.6772156, -435.7330017, 437.7506714
1: -199.7766724, 167.9270020, -207.8321228, 174.6739044, -374.4505615, 375.7590942
2: -262.5853577, 170.4111481, -273.1289368, 177.1750793, -439.7604065, 443.5400391
3: -279.3592834, 147.4691925, -290.5870361, 153.3434601, -432.7027283, 438.0562134
4: -256.2437134, 196.3322906, -266.5183105, 204.2113647, -460.4550781, 462.8505859
5: -229.6284485, 178.4928589, -238.8408508, 185.6314087, -415.2598572, 417.3335571
6: -219.7420044, 210.8782043, -228.5142517, 219.3156281, -439.0576172, 439.3924255
7: -239.1328278, 200.9109955, -248.7226715, 208.8996124, -448.0324097, 449.6336670
8: -288.1436157, 196.6213989, -299.6381226, 204.4494934, -492.5931091, 496.2595215
9: -217.6873322, 214.2570953, -226.3706665, 222.8048248, -440.4921570, 440.6277466

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2208157, upper bound: 461.2215481
time: 9.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2164158, upper bound: 461.2158329
time: 10.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -242.0653381, 191.5312653, -248.6029663, 196.6772156, -438.7425537, 440.1342163
1: -202.2512054, 170.0212555, -207.8321228, 174.6739044, -376.9251099, 377.8533630
2: -265.8876343, 172.5388794, -273.1289368, 177.1750793, -443.0626831, 445.6678162
3: -282.8842773, 149.2758636, -290.5870361, 153.3434601, -436.2277222, 439.8629150
4: -259.4489136, 198.7559357, -266.5183105, 204.2113647, -463.6602783, 465.2742310
5: -232.4912720, 180.6936493, -238.8408508, 185.6314087, -418.1226807, 419.5344238
6: -222.5059814, 213.5375061, -228.5142517, 219.3156281, -441.8215942, 442.0516968
7: -242.1423340, 203.4313660, -248.7226715, 208.8996124, -451.0418701, 452.1540222
8: -291.7273254, 199.0342255, -299.6381226, 204.4494934, -496.1768188, 498.6723328
9: -220.4096527, 216.9235992, -226.3706665, 222.8048248, -443.2144775, 443.2942505

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2256253, upper bound: 461.2251267
time: 8.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259528, upper bound: 461.2259528
time: 8.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.62 seconds
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2224367, upper bound: 461.2213545
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2205328, upper bound: 461.2184812
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2220055, upper bound: 461.2197122
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2225099, upper bound: 461.2209312
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2208157, upper bound: 461.2215481
IS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2164158, upper bound: 461.2158329
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2256253, upper bound: 461.2251267
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 7, lower bound: -461.2259528, upper bound: 461.2259528

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -237.1957703, 187.6796570, -243.2792664, 192.4492188, -429.6449890, 430.9588623
1: -198.2167969, 166.6274414, -203.3293762, 170.8664246, -369.0831909, 369.9568176
2: -260.5465088, 169.0961456, -267.2058411, 173.3429260, -433.8894043, 436.3020020
3: -277.2070312, 146.3290863, -284.2699280, 150.0049133, -427.2119446, 430.5989990
4: -254.2576904, 194.8118591, -260.6769104, 199.7044525, -453.9621582, 455.4887390
5: -227.8454895, 177.1158905, -233.6641235, 181.4764557, -409.3218689, 410.7800293
6: -218.0363464, 209.2415771, -223.5851593, 214.5607910, -432.5971375, 432.8267212
7: -237.2802124, 199.3645020, -243.2903290, 204.3175812, -441.5977783, 442.6548157
8: -285.9001465, 195.0919037, -293.1693420, 199.9959869, -485.8960571, 488.2612000
9: -216.0109711, 212.5939484, -221.3304138, 217.9026031, -433.9135742, 433.9242554

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2200949, upper bound: 461.2177210
time: 10.04 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2207685, upper bound: 461.2191418
time: 10.54 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -237.0844269, 187.5917969, -247.6648102, 195.9571533, -433.0415649, 435.2565918
1: -198.1233368, 166.5492401, -206.9805756, 173.9261322, -372.0494690, 373.5298157
2: -260.4240112, 169.0173950, -272.0266724, 176.4197083, -436.8437195, 441.0440063
3: -277.0773315, 146.2602997, -289.4729919, 152.6017303, -429.6790771, 435.7332764
4: -254.1381378, 194.7201843, -265.3915710, 203.2683258, -457.4064636, 460.1117554
5: -227.7384796, 177.0328369, -237.8928528, 184.7036285, -412.4421082, 414.9256897
6: -217.9342651, 209.1433411, -227.6363983, 218.4482117, -436.3824768, 436.7797241
7: -237.1689301, 199.2714233, -247.6827850, 207.9847565, -445.1536865, 446.9541931
8: -285.7656555, 194.9996490, -298.3940735, 203.4707336, -489.2363586, 493.3937378
9: -215.9098511, 212.4940033, -225.3219604, 221.8156281, -437.7254333, 437.8159790

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1860511, upper bound: 461.2154461
time: 11.24 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2187776, upper bound: 461.2159985
time: 10.75 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -240.2525635, 190.1009216, -241.4315338, 190.9862518, -431.2388306, 431.5324097
1: -200.7316284, 168.7547913, -201.7806854, 169.5434113, -370.2750244, 370.5354614
2: -263.9010315, 171.2569733, -265.1640015, 171.9996643, -435.9006958, 436.4209290
3: -280.7868042, 148.1636963, -282.0538940, 148.8870087, -429.6738281, 430.2175598
4: -257.5139160, 197.2751007, -258.6121826, 198.1743164, -455.6882324, 455.8872681
5: -230.7540436, 179.3523102, -231.8826904, 180.0371246, -410.7911682, 411.2349854
6: -220.8434448, 211.9424744, -221.8820953, 212.9161530, -433.7595825, 433.8245850
7: -240.3375397, 201.9243011, -241.3613739, 202.7235260, -443.0610657, 443.2856750
8: -289.5400085, 197.5443268, -290.9928894, 198.5123444, -488.0523682, 488.5372314
9: -218.7767639, 215.3033447, -219.5595551, 216.2307129, -435.0074768, 434.8629150

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2091512, upper bound: 461.2036775
time: 9.21 seconds

## Relational analysis of IS_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2035812, upper bound: 461.1995315
time: 9.19 seconds

## BFS IS instance: IS_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -240.2525635, 190.1009216, -244.5962524, 193.4879150, -433.7404785, 434.6971741
1: -200.7316284, 168.7547913, -204.4331207, 171.7718353, -372.5034180, 373.1879272
2: -263.9010315, 171.2569733, -268.6480103, 174.2654419, -438.1664734, 439.9049377
3: -280.7868042, 148.1636963, -285.7642822, 150.8323975, -431.6192017, 433.9279480
4: -257.5139160, 197.2751007, -262.0426025, 200.7837677, -458.2976685, 459.3176880
5: -230.7540436, 179.3523102, -234.9331970, 182.4174957, -413.1714478, 414.2854919
6: -220.8434448, 211.9424744, -224.7886963, 215.7072754, -436.5507202, 436.7311401
7: -240.3375397, 201.9243011, -244.5552521, 205.3948364, -445.7323608, 446.4795532
8: -289.5400085, 197.5443268, -294.7833862, 201.1089630, -490.6489563, 492.3276978
9: -218.7767639, 215.3033447, -222.4762115, 219.0664673, -437.8431396, 437.7795410

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2123602, upper bound: 461.2125868
time: 10.19 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2058085, upper bound: 461.2033412
time: 9.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -239.0557861, 189.1477356, -246.7524872, 195.2156982, -434.2714539, 435.9001770
1: -199.7766724, 167.9270020, -206.2854309, 173.3769836, -373.1536560, 374.2124329
2: -262.5853577, 170.4111481, -271.0966492, 175.8600464, -438.4454041, 441.5077209
3: -279.3592834, 147.4691925, -288.4180298, 152.2021790, -431.5614624, 435.8872070
4: -256.2437134, 196.3322906, -264.5209045, 202.6973114, -458.9409485, 460.8531799
5: -229.6284485, 178.4928589, -237.0634613, 184.2507477, -413.8792114, 415.5562439
6: -219.7420044, 210.8782043, -226.8096771, 217.6835480, -437.4255371, 437.6878662
7: -239.1328278, 200.9109955, -246.8627014, 207.3440704, -446.4768982, 447.7736816
8: -288.1436157, 196.6213989, -297.4191895, 202.9339294, -491.0775146, 494.0405884
9: -217.6873322, 214.2570953, -224.6814575, 221.1537018, -438.8410034, 438.9385376

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2194259, upper bound: 461.2195127
time: 9.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2197072, upper bound: 461.2204360
time: 10.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -242.0653381, 191.5312653, -241.9723206, 191.4354095, -433.5007324, 433.5036011
1: -202.2512054, 170.0212555, -202.2716217, 169.9871521, -372.2383423, 372.2928772
2: -265.8876343, 172.5388794, -265.8232117, 172.4467621, -438.3343811, 438.3620911
3: -282.8842773, 149.2758636, -282.8073120, 149.2766418, -432.1609192, 432.0831604
4: -259.4489136, 198.7559357, -259.3140564, 198.7366180, -458.1855469, 458.0700073
5: -232.4912720, 180.6936493, -232.4613190, 180.6084747, -413.0997009, 413.1549683
6: -222.5059814, 213.5375061, -222.4272308, 213.4507141, -435.9566956, 435.9647217
7: -242.1423340, 203.4313660, -242.0118866, 203.3056335, -445.4479370, 445.4432373
8: -291.7273254, 199.0342255, -291.6854858, 198.9967346, -490.7240601, 490.7196960
9: -220.4096527, 216.9235992, -220.2518158, 216.8367767, -437.2463989, 437.1754150

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2238513, upper bound: 461.2236497
time: 8.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2230799, upper bound: 461.2223405
time: 9.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -242.0653381, 191.5312653, -244.6914215, 193.5844421, -435.6497803, 436.2226868
1: -202.2512054, 170.0212555, -204.5505981, 171.9045715, -374.1557617, 374.5718384
2: -265.8876343, 172.5388794, -268.8150024, 174.3938904, -440.2814636, 441.3538513
3: -282.8842773, 149.2758636, -285.9944458, 150.9472198, -433.8314819, 435.2703247
4: -259.4489136, 198.7559357, -262.2655334, 200.9803467, -460.4292603, 461.0214844
5: -232.4912720, 180.6936493, -235.0823059, 182.6605072, -415.1517639, 415.7759399
6: -222.5059814, 213.5375061, -224.9223022, 215.8472137, -438.3532104, 438.4597778
7: -242.1423340, 203.4313660, -244.7566071, 205.6009827, -447.7432861, 448.1879578
8: -291.7273254, 199.0342255, -294.9316711, 201.2245178, -492.9517822, 493.9658813
9: -220.4096527, 216.9235992, -222.7602539, 219.2725830, -439.6822510, 439.6838379

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2241010, upper bound: 461.2245109
time: 9.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2232565, upper bound: 461.2232565
time: 8.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.97 seconds
IS_A2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2200949, upper bound: 461.2177210
IS_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2207685, upper bound: 461.2191418
IS_A2_B1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.1860511, upper bound: 461.2154461
IS_A2_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2187776, upper bound: 461.2159985
IS_A2_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2091512, upper bound: 461.2036775
IS_A2_B1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2035812, upper bound: 461.1995315
IS_A2_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2123602, upper bound: 461.2125868
IS_A2_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2058085, upper bound: 461.2033412
IS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2194259, upper bound: 461.2195127
IS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2197072, upper bound: 461.2204360
IS_A2_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2238513, upper bound: 461.2236497
IS_A2_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2230799, upper bound: 461.2223405
IS_A2_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2241010, upper bound: 461.2245109
IS_A2_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 18.97
Output dim: 7, lower bound: -461.2232565, upper bound: 461.2232565

## BFS IS instance: IS_A2_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -237.1957703, 187.6796570, -236.3219604, 186.9501190, -424.1458740, 424.0015259
1: -198.2167969, 166.6274414, -197.4952850, 165.9524384, -364.1691895, 364.1226807
2: -260.5465088, 169.0961456, -259.5387268, 168.3805237, -428.9269714, 428.6348877
3: -277.2070312, 146.3290863, -276.1054688, 145.7362213, -422.9432373, 422.4345703
4: -254.2576904, 194.8118591, -253.1203461, 193.9617310, -448.2194214, 447.9321594
5: -227.8454895, 177.1158905, -226.9717560, 176.2183990, -404.0638733, 404.0876465
6: -218.0363464, 209.2415771, -217.1944580, 208.4063568, -426.4426880, 426.4360352
7: -237.2802124, 199.3645020, -236.2492981, 198.4480438, -435.7282715, 435.6138000
8: -285.9001465, 195.0919037, -284.8167419, 194.2719574, -480.1720276, 479.9085999
9: -216.0109711, 212.5939484, -214.9116974, 211.6424713, -427.6534424, 427.5055847

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2059048, upper bound: 461.2008543
time: 9.54 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2009051, upper bound: 461.1972186
time: 9.36 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -237.1957703, 187.6796570, -239.5165253, 189.4734802, -426.6692505, 427.1961670
1: -198.2167969, 166.6274414, -200.1723328, 168.2014618, -366.4182434, 366.7997742
2: -260.5465088, 169.0961456, -263.0543213, 170.6664124, -431.2128906, 432.1504517
3: -277.2070312, 146.3290863, -279.8515320, 147.6993713, -424.9064026, 426.1806030
4: -254.2576904, 194.8118591, -256.5814819, 196.5952911, -450.8529663, 451.3933411
5: -227.8454895, 177.1158905, -230.0495758, 178.6198425, -406.4653320, 407.1654358
6: -218.0363464, 209.2415771, -220.1275330, 211.2229919, -429.2593384, 429.3691101
7: -237.2802124, 199.3645020, -239.4718018, 201.1438904, -438.4241028, 438.8362732
8: -285.9001465, 195.0919037, -288.6414185, 196.8925934, -482.7927246, 483.7333069
9: -216.0109711, 212.5939484, -217.8559113, 214.5037384, -430.5146790, 430.4497986

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2088624, upper bound: 461.2050186
time: 9.58 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2040768, upper bound: 461.2014723
time: 10.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -239.0557861, 189.1477356, -240.0894623, 189.9488220, -429.0045776, 429.2371826
1: -199.7766724, 167.9270020, -200.6984406, 168.6682434, -368.4449158, 368.6254272
2: -262.5853577, 170.4111481, -263.7560120, 171.1086273, -433.6939392, 434.1670837
3: -279.3592834, 147.4691925, -280.6012878, 148.1158600, -427.4750671, 428.0704956
4: -256.2437134, 196.3322906, -257.2824097, 197.1966248, -453.4403381, 453.6146851
5: -229.6284485, 178.4928589, -230.6532593, 179.2045441, -408.8330078, 409.1459961
6: -219.7420044, 210.8782043, -220.6931610, 211.7908783, -431.5328674, 431.5713501
7: -239.1328278, 200.9109955, -240.1201935, 201.7235107, -440.8563232, 441.0311584
8: -288.1436157, 196.6213989, -289.4283142, 197.4555817, -485.5991821, 486.0497131
9: -217.6873322, 214.2570953, -218.5338287, 215.1573944, -432.8447266, 432.7909241

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2175178, upper bound: 461.2171568
time: 9.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2153695, upper bound: 461.2157214
time: 9.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -239.0557861, 189.1477356, -242.8328247, 192.1167755, -431.1725159, 431.9804688
1: -199.7766724, 167.9270020, -202.9973755, 170.6023102, -370.3789673, 370.9243469
2: -262.5853577, 170.4111481, -266.7741699, 173.0730286, -435.6583862, 437.1852417
3: -279.3592834, 147.4691925, -283.8164978, 149.8012085, -429.1604614, 431.2857056
4: -256.2437134, 196.3322906, -260.2593994, 199.4599457, -455.7036438, 456.5916748
5: -229.6284485, 178.4928589, -233.2973785, 181.2741394, -410.9025879, 411.7901917
6: -219.7420044, 210.8782043, -223.2103119, 214.2083740, -433.9503174, 434.0884399
7: -239.1328278, 200.9109955, -242.8887024, 204.0388947, -443.1717224, 443.7996826
8: -288.1436157, 196.6213989, -292.7033997, 199.7027283, -487.8463135, 489.3247986
9: -217.6873322, 214.2570953, -221.0639801, 217.6147156, -435.3020020, 435.3210754

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2163005, upper bound: 461.2178785
time: 9.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2156804, upper bound: 461.2167624
time: 7.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -242.0653381, 191.5312653, -236.8150940, 187.3616943, -429.4270020, 428.3463745
1: -202.2512054, 170.0212555, -197.9470215, 166.3628235, -368.6140137, 367.9682617
2: -265.8876343, 172.5388794, -260.1445007, 168.7944641, -434.6820984, 432.6833801
3: -282.8842773, 149.2758636, -276.8014832, 146.0958862, -428.9801331, 426.0773315
4: -259.4489136, 198.7559357, -253.7707520, 194.4860229, -453.9349365, 452.5266724
5: -232.4912720, 180.6936493, -227.5041351, 176.7547302, -409.2459412, 408.1977844
6: -222.5059814, 213.5375061, -217.6949921, 208.8984833, -431.4044800, 431.2324219
7: -242.1423340, 203.4313660, -236.8507233, 198.9903564, -441.1326599, 440.2821045
8: -291.7273254, 199.0342255, -285.4518738, 194.7183228, -486.4456482, 484.4860840
9: -220.4096527, 216.9235992, -215.5604401, 212.2049866, -432.6146240, 432.4840393

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2146131, upper bound: 461.2119573
time: 8.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2084514, upper bound: 461.2075652
time: 8.22 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -241.9603882, 191.4483795, -241.2981262, 190.9405670, -432.9009094, 432.7464905
1: -202.1630859, 169.9474945, -201.6884003, 169.4938507, -371.6569214, 371.6358643
2: -265.7720947, 172.4645691, -265.0829773, 171.9436951, -437.7157593, 437.5475464
3: -282.7620850, 149.2109375, -282.1284790, 148.7567444, -431.5188293, 431.3394165
4: -259.3361511, 198.6694794, -258.6002197, 198.1357727, -457.4718933, 457.2696838
5: -232.3904266, 180.6152954, -231.8247070, 180.0437927, -412.4341431, 412.4399719
6: -222.4097290, 213.4448395, -221.8442078, 212.8800354, -435.2897644, 435.2889709
7: -242.0373688, 203.3436279, -241.3505249, 202.7423401, -444.7796936, 444.6940918
8: -291.6003723, 198.9470978, -290.8102722, 198.2771454, -489.8775024, 489.7573853
9: -220.3142548, 216.8293610, -219.6469421, 216.2089844, -436.5232544, 436.4763184

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2135704, upper bound: 461.2102419
time: 11.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2074116, upper bound: 461.2059387
time: 10.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -242.0653381, 191.5312653, -239.5440216, 189.5166626, -431.5820007, 431.0752563
1: -202.2512054, 170.0212555, -200.2338104, 168.2868195, -370.5379944, 370.2550354
2: -265.8876343, 172.5388794, -263.1466980, 170.7478180, -436.6354370, 435.6855164
3: -282.8842773, 149.2758636, -280.0006409, 147.7723083, -430.6565857, 429.2764893
4: -259.4489136, 198.7559357, -256.7309875, 196.7366333, -456.1855469, 455.4869080
5: -232.4912720, 180.6936493, -230.1337891, 178.8123932, -411.3036499, 410.8274536
6: -222.5059814, 213.5375061, -220.1987152, 211.3031311, -433.8091125, 433.7362061
7: -242.1423340, 203.4313660, -239.6041870, 201.2930298, -443.4352722, 443.0355225
8: -291.7273254, 199.0342255, -288.7082520, 196.9533234, -488.6806335, 487.7424622
9: -220.4096527, 216.9235992, -218.0763550, 214.6478424, -435.0574951, 434.9999390

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2137612, upper bound: 461.2161276
time: 11.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2095512, upper bound: 461.2101354
time: 9.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -241.9603882, 191.4483795, -244.0984802, 193.1538544, -435.1141968, 435.5468750
1: -202.1630859, 169.9474945, -204.0256500, 171.4624176, -373.6254883, 373.9730835
2: -265.7720947, 172.4645691, -268.1503906, 173.9402161, -439.7122803, 440.6149597
3: -282.7620850, 149.2109375, -285.3977661, 150.4680634, -433.2301636, 434.6087036
4: -259.3361511, 198.6694794, -261.6254578, 200.4368134, -459.7729492, 460.2949219
5: -232.3904266, 180.6152954, -234.5225677, 182.1636047, -414.5539856, 415.1378174
6: -222.4097290, 213.4448395, -224.4008331, 215.3375702, -437.7473145, 437.8456726
7: -242.0373688, 203.3436279, -244.1631470, 205.0982819, -447.1355896, 447.5067139
8: -291.6003723, 198.9470978, -294.1322327, 200.5664368, -492.1668091, 493.0793457
9: -220.3142548, 216.8293610, -222.2199554, 218.7087860, -439.0230408, 439.0493164

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2149449, upper bound: 461.2130403
time: 10.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2085852, upper bound: 461.2085852
time: 7.93 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 19.62 seconds
IS_A2_B1_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2059048, upper bound: 461.2008543
IS_A2_B1_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2009051, upper bound: 461.1972186
IS_A2_B1_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2088624, upper bound: 461.2050186
IS_A2_B1_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2040768, upper bound: 461.2014723
IS_A2_B2_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2175178, upper bound: 461.2171568
IS_A2_B2_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2153695, upper bound: 461.2157214
IS_A2_B2_A1_B2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2163005, upper bound: 461.2178785
IS_A2_B2_A1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2156804, upper bound: 461.2167624
IS_A2_B2_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2146131, upper bound: 461.2119573
IS_A2_B2_A2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2084514, upper bound: 461.2075652
IS_A2_B2_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2135704, upper bound: 461.2102419
IS_A2_B2_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2074116, upper bound: 461.2059387
IS_A2_B2_A2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2137612, upper bound: 461.2161276
IS_A2_B2_A2_B2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2095512, upper bound: 461.2101354
IS_A2_B2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2149449, upper bound: 461.2130403
IS_A2_B2_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 19.62
Output dim: 7, lower bound: -461.2085852, upper bound: 461.2085852
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=464.3514404296875
rel_dist={7: [-461.2313563734555, 461.2313561033127]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2148347, upper bound: 461.2202370
time: 7.95 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2313740, upper bound: 461.2313740
time: 7.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.61
Output dim: 7, lower bound: -461.2148347, upper bound: 461.2202370
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.61
Output dim: 7, lower bound: -461.2313740, upper bound: 461.2313740

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -262.5196838, 207.7006226, -252.2369537, 199.5569458, -462.0765991, 459.9375610
1: -219.5079498, 184.4597931, -210.8871155, 177.2368164, -396.7446899, 395.3468933
2: -288.4674072, 187.0221405, -277.1375732, 179.7647705, -468.2321167, 464.1597290
3: -306.9580994, 161.8770752, -294.8458557, 155.5839844, -462.5420837, 456.7229309
4: -281.4533691, 215.6667480, -270.4437866, 207.2173462, -488.6707153, 486.1105347
5: -252.2645111, 196.0215607, -242.3396301, 188.3740387, -440.6385498, 438.3612061
6: -241.2943268, 231.6137390, -231.8667755, 222.5341949, -463.8284302, 463.4804688
7: -262.6724548, 220.5553284, -252.3775177, 211.9631805, -474.6356201, 472.9328613
8: -316.4441833, 215.8687744, -304.0287170, 207.4399414, -523.8841553, 519.8973999
9: -239.0110474, 235.2242737, -229.7071686, 226.0893707, -465.1004028, 464.9314575

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2108213, upper bound: 461.2160528
time: 10.05 seconds

## Relational analysis of IS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2132193, upper bound: 461.2176896
time: 8.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2148207, upper bound: 461.2202337
time: 9.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -252.2428284, 199.5615692, -451.7855835, 451.7895203
1: -210.8763428, 177.2277527, -210.8920135, 177.2409058, -388.1171875, 388.1196899
2: -277.1233826, 179.7556458, -277.1440125, 179.7689209, -456.8922729, 456.8995667
3: -294.8307190, 155.5760193, -294.8527222, 155.5875702, -450.4182739, 450.4287109
4: -270.4299011, 207.2068634, -270.4500427, 207.2221527, -477.6520386, 477.6569214
5: -242.3272858, 188.3644562, -242.3452606, 188.3783875, -430.7056580, 430.7096863
6: -231.8549500, 222.5227814, -231.8721466, 222.5393829, -454.3942871, 454.3949280
7: -252.3645477, 211.9523926, -252.3833771, 211.9680939, -464.3326111, 464.3357544
8: -304.0132751, 207.4294128, -304.0357361, 207.4447021, -511.4579773, 511.4651489
9: -229.6954498, 226.0778656, -229.7125092, 226.0945892, -455.7900391, 455.7903442

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2276929, upper bound: 461.2279638
time: 8.41 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271599
time: 8.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.15 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 18.15
Output dim: 7, lower bound: -461.2132193, upper bound: 461.2176896
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.15
Output dim: 7, lower bound: -461.2148207, upper bound: 461.2202337
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.15
Output dim: 7, lower bound: -461.2276929, upper bound: 461.2279638
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.15
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271599

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -262.5196838, 207.7006226, -250.1860504, 197.9248962, -460.4445801, 457.8866577
1: -219.5079498, 184.4597931, -209.1587372, 175.7771759, -395.2850952, 393.6185303
2: -288.4674072, 187.0221405, -274.8612061, 178.2923126, -466.7596741, 461.8833618
3: -306.9580994, 161.8770752, -292.4113770, 154.3120880, -461.2702026, 454.2884521
4: -281.4533691, 215.6667480, -268.2033081, 205.5008087, -486.9541626, 483.8700562
5: -252.2645111, 196.0215607, -240.3562164, 186.7968750, -439.0614014, 436.3777771
6: -241.2943268, 231.6137390, -229.9635162, 220.7060242, -462.0002747, 461.5772400
7: -262.6724548, 220.5553284, -250.2947693, 210.2102814, -472.8827515, 470.8500977
8: -316.4441833, 215.8687744, -301.5472107, 205.7504883, -522.1945801, 517.4159546
9: -239.0110474, 235.2242737, -227.7890778, 224.2151794, -463.2261658, 463.0133667

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2107369, upper bound: 461.2160109
time: 9.83 seconds

## Relational analysis of IS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2116332, upper bound: 461.2176642
time: 7.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2135562, upper bound: 461.2190642
time: 8.15 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -239.0743866, 189.1624146, -441.3864441, 438.6211243
1: -210.8763428, 177.2277527, -199.7922211, 167.9400635, -378.8164062, 377.0199280
2: -277.1233826, 179.7556458, -262.6058350, 170.4243469, -447.5476990, 442.3613892
3: -294.8307190, 155.5760193, -279.3811340, 147.4806366, -442.3113403, 434.9571228
4: -270.4299011, 207.2068634, -256.2637939, 196.3474579, -466.7773438, 463.4706421
5: -242.3272858, 188.3644562, -229.6463013, 178.5066833, -420.8338623, 418.0107422
6: -231.8549500, 222.5227814, -219.7590637, 210.8946838, -442.7495728, 442.2818298
7: -252.3645477, 211.9523926, -239.1515045, 200.9265594, -453.2911072, 451.1038818
8: -304.0132751, 207.4294128, -288.1658630, 196.6365509, -500.6498413, 495.5952759
9: -229.6954498, 226.0778656, -217.7042542, 214.2737274, -443.9691467, 443.7821045

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2238276, upper bound: 461.2253166
time: 9.73 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2276836, upper bound: 461.2279305
time: 10.57 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -242.0842743, 191.5462341, -443.7702332, 441.6310120
1: -210.8763428, 177.2277527, -202.2670288, 170.0345459, -380.9108887, 379.4947510
2: -277.1233826, 179.7556458, -265.9084778, 172.5523376, -449.6757202, 445.6640320
3: -294.8307190, 155.5760193, -282.9065552, 149.2875214, -444.1182251, 438.4825439
4: -270.4299011, 207.2068634, -259.4692688, 198.7713928, -469.2012939, 466.6761475
5: -242.3272858, 188.3644562, -232.5094452, 180.7077179, -423.0350037, 420.8738708
6: -231.8549500, 222.5227814, -222.5233765, 213.5542908, -445.4092102, 445.0461426
7: -252.3645477, 211.9523926, -242.1613312, 203.4472351, -455.8117065, 454.1136780
8: -304.0132751, 207.4294128, -291.7500916, 199.0496979, -503.0629883, 499.1795044
9: -229.6954498, 226.0778656, -220.4269104, 216.9405365, -446.6359863, 446.5047607

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2232979, upper bound: 461.2245496
time: 9.18 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271545, upper bound: 461.2271545
time: 8.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.08 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 19.08
Output dim: 7, lower bound: -461.2116332, upper bound: 461.2176642
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 19.08
Output dim: 7, lower bound: -461.2135562, upper bound: 461.2190642
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.08
Output dim: 7, lower bound: -461.2238276, upper bound: 461.2253166
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.08
Output dim: 7, lower bound: -461.2276836, upper bound: 461.2279305
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.08
Output dim: 7, lower bound: -461.2232979, upper bound: 461.2245496
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.08
Output dim: 7, lower bound: -461.2271545, upper bound: 461.2271545

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -248.3659973, 196.4675598, -239.0743866, 189.1624146, -437.5284119, 435.5419312
1: -207.5953979, 174.4416504, -199.7922211, 167.9400635, -375.5354004, 374.2338867
2: -272.8066406, 176.9463654, -262.6058350, 170.4243469, -443.2309570, 439.5520935
3: -290.1911926, 153.1420135, -279.3811340, 147.4806366, -437.6718140, 432.5231323
4: -266.1442566, 203.8983154, -256.2637939, 196.3474579, -462.4916992, 460.1620789
5: -238.5536957, 185.2788849, -229.6463013, 178.5066833, -417.0603333, 414.9251709
6: -228.2519379, 219.0505829, -219.7590637, 210.8946838, -439.1465759, 438.8096313
7: -248.3798370, 208.5741119, -239.1515045, 200.9265594, -449.3063660, 447.7255859
8: -299.3194885, 204.2174377, -288.1658630, 196.6365509, -495.9560547, 492.3833008
9: -225.9566650, 222.4711304, -217.7042542, 214.2737274, -440.2304077, 440.1753845

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2206654, upper bound: 461.2234293
time: 9.45 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2218927, upper bound: 461.2238378
time: 8.63 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -250.1731720, 197.9147339, -239.0743866, 189.1624146, -439.3355408, 436.9891357
1: -209.1479492, 175.7681427, -199.7922211, 167.9400635, -377.0880127, 375.5603638
2: -274.8470154, 178.2832031, -262.6058350, 170.4243469, -445.2713318, 440.8890076
3: -292.3962097, 154.3041382, -279.3811340, 147.4806366, -439.8768311, 433.6852417
4: -268.1894531, 205.4903107, -256.2637939, 196.3474579, -464.5369263, 461.7540894
5: -240.3438568, 186.7873077, -229.6463013, 178.5066833, -418.8504944, 416.4335938
6: -229.9516754, 220.6946259, -219.7590637, 210.8946838, -440.8463440, 440.4536438
7: -250.2817993, 210.1995087, -239.1515045, 200.9265594, -451.2083740, 449.3510132
8: -301.5317688, 205.7399750, -288.1658630, 196.6365509, -498.1683350, 493.9058228
9: -227.7773590, 224.2036438, -217.7042542, 214.2737274, -442.0510864, 441.9078674

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2257208, upper bound: 461.2265453
time: 8.23 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2265406, upper bound: 461.2268540
time: 8.38 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -248.3659973, 196.4675598, -242.0842743, 191.5462341, -439.9122314, 438.5518188
1: -207.5953979, 174.4416504, -202.2670288, 170.0345459, -377.6299133, 376.7086792
2: -272.8066406, 176.9463654, -265.9084778, 172.5523376, -445.3589478, 442.8547668
3: -290.1911926, 153.1420135, -282.9065552, 149.2875214, -439.4786987, 436.0485840
4: -266.1442566, 203.8983154, -259.4692688, 198.7713928, -464.9156494, 463.3675842
5: -238.5536957, 185.2788849, -232.5094452, 180.7077179, -419.2614136, 417.7882996
6: -228.2519379, 219.0505829, -222.5233765, 213.5542908, -441.8062134, 441.5739746
7: -248.3798370, 208.5741119, -242.1613312, 203.4472351, -451.8269958, 450.7353821
8: -299.3194885, 204.2174377, -291.7500916, 199.0496979, -498.3691711, 495.9675293
9: -225.9566650, 222.4711304, -220.4269104, 216.9405365, -442.8972168, 442.8980408

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2200899, upper bound: 461.2225511
time: 8.95 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2213190, upper bound: 461.2229692
time: 8.88 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -250.1731720, 197.9147339, -242.0842743, 191.5462341, -441.7193298, 439.9990234
1: -209.1479492, 175.7681427, -202.2670288, 170.0345459, -379.1824951, 378.0351562
2: -274.8470154, 178.2832031, -265.9084778, 172.5523376, -447.3993225, 444.1916504
3: -292.3962097, 154.3041382, -282.9065552, 149.2875214, -441.6837158, 437.2106934
4: -268.1894531, 205.4903107, -259.4692688, 198.7713928, -466.9608459, 464.9595947
5: -240.3438568, 186.7873077, -232.5094452, 180.7077179, -421.0515747, 419.2967529
6: -229.9516754, 220.6946259, -222.5233765, 213.5542908, -443.5059509, 443.2180176
7: -250.2817993, 210.1995087, -242.1613312, 203.4472351, -453.7290344, 452.3607788
8: -301.5317688, 205.7399750, -291.7500916, 199.0496979, -500.5814819, 497.4900513
9: -227.7773590, 224.2036438, -220.4269104, 216.9405365, -444.7178955, 444.6305237

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251910, upper bound: 461.2256916
time: 7.98 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259905
time: 9.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.24 seconds
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2206654, upper bound: 461.2234293
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2218927, upper bound: 461.2238378
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2257208, upper bound: 461.2265453
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2265406, upper bound: 461.2268540
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2200899, upper bound: 461.2225511
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2213190, upper bound: 461.2229692
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2251910, upper bound: 461.2256916
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 18.24
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259905

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -239.0743866, 189.1624146, -430.5752563, 430.0458984
1: -201.7650757, 169.5303040, -199.7922211, 167.9400635, -369.7051392, 369.3225098
2: -265.1434326, 171.9864349, -262.6058350, 170.4243469, -435.5677490, 434.5922852
3: -282.0320129, 148.8755341, -279.3811340, 147.4806366, -429.5126343, 428.2566528
4: -258.5920715, 198.1590881, -256.2637939, 196.3474579, -454.9395142, 454.4228516
5: -231.8648071, 180.0232697, -229.6463013, 178.5066833, -410.3713989, 409.6695557
6: -221.8649445, 212.8996429, -219.7590637, 210.8946838, -432.7596130, 432.6586609
7: -241.3426514, 202.7079010, -239.1515045, 200.9265594, -442.2691956, 441.8594055
8: -290.9705505, 198.4971466, -288.1658630, 196.6365509, -487.6071167, 486.6630249
9: -219.5425873, 216.2140350, -217.7042542, 214.2737274, -433.8163147, 433.9182739

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2107724, upper bound: 461.2123965
time: 9.17 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2206654, upper bound: 461.2234293
time: 9.43 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -244.5776367, 193.4732056, -239.0743866, 189.1624146, -433.7400513, 432.5476074
1: -204.4175262, 171.7587585, -199.7922211, 167.9400635, -372.3575745, 371.5509644
2: -268.6275330, 174.2521973, -262.6058350, 170.4243469, -439.0518494, 436.8580322
3: -285.7423706, 150.8209229, -279.3811340, 147.4806366, -433.2230225, 430.2019958
4: -262.0225525, 200.7685699, -256.2637939, 196.3474579, -458.3699951, 457.0323486
5: -234.9153442, 182.4036560, -229.6463013, 178.5066833, -413.4219360, 412.0499573
6: -224.7715912, 215.6907806, -219.7590637, 210.8946838, -435.6661987, 435.4498291
7: -244.5365143, 205.3792267, -239.1515045, 200.9265594, -445.4630737, 444.5307312
8: -294.7610779, 201.0937653, -288.1658630, 196.6365509, -491.3976440, 489.2596130
9: -222.4592285, 219.0498047, -217.7042542, 214.2737274, -436.7329407, 436.7540283

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2121159, upper bound: 461.2128675
time: 11.83 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2218927, upper bound: 461.2238378
time: 9.80 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -243.5448151, 192.6747894, -239.0743866, 189.1624146, -432.7071838, 431.7491760
1: -203.5896149, 171.0831757, -199.7922211, 167.9400635, -371.5296631, 370.8753967
2: -267.5437622, 173.5565033, -262.6058350, 170.4243469, -437.9680786, 436.1623230
3: -284.6197510, 150.2388916, -279.3811340, 147.4806366, -432.1004028, 429.6199646
4: -260.9874573, 200.0178375, -256.2637939, 196.3474579, -457.3348999, 456.2816162
5: -233.9668427, 181.7662506, -229.6463013, 178.5066833, -412.4734497, 411.4125366
6: -223.8668671, 214.8315582, -219.7590637, 210.8946838, -434.7615356, 434.5906372
7: -243.5734253, 204.6076660, -239.1515045, 200.9265594, -444.5000000, 443.7591553
8: -293.5816956, 200.2887726, -288.1658630, 196.6365509, -490.2182617, 488.4546509
9: -221.6605225, 218.2372437, -217.7042542, 214.2737274, -435.9342346, 435.9414673

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2055152, upper bound: 461.2092504
time: 11.34 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2257208, upper bound: 461.2265453
time: 8.74 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -246.2702332, 194.8287354, -239.0743866, 189.1624146, -435.4326477, 433.9031067
1: -205.8732300, 173.0046844, -199.7922211, 167.9400635, -373.8132629, 372.7969055
2: -270.5419922, 175.5079803, -262.6058350, 170.4243469, -440.9663086, 438.1137695
3: -287.8132019, 151.9130249, -279.3811340, 147.4806366, -435.2938232, 431.2940979
4: -263.9453735, 202.2661743, -256.2637939, 196.3474579, -460.2928467, 458.5299377
5: -236.5937195, 183.8229675, -229.6463013, 178.5066833, -415.1003723, 413.4692688
6: -226.3674469, 217.2335510, -219.7590637, 210.8946838, -437.2621155, 436.9926147
7: -246.3242340, 206.9079742, -239.1515045, 200.9265594, -447.2507935, 446.0594177
8: -296.8351135, 202.5220795, -288.1658630, 196.6365509, -493.4716797, 490.6879272
9: -224.1749573, 220.6789398, -217.7042542, 214.2737274, -438.4486694, 438.3831787

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2098006, upper bound: 461.2121421
time: 9.50 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2265406, upper bound: 461.2268540
time: 8.02 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -242.0842743, 191.5462341, -432.9590759, 433.0557861
1: -201.7650757, 169.5303040, -202.2670288, 170.0345459, -371.7996216, 371.7973328
2: -265.1434326, 171.9864349, -265.9084778, 172.5523376, -437.6957092, 437.8948975
3: -282.0320129, 148.8755341, -282.9065552, 149.2875214, -431.3195190, 431.7821045
4: -258.5920715, 198.1590881, -259.4692688, 198.7713928, -457.3634644, 457.6283569
5: -231.8648071, 180.0232697, -232.5094452, 180.7077179, -412.5725098, 412.5327148
6: -221.8649445, 212.8996429, -222.5233765, 213.5542908, -435.4192505, 435.4230347
7: -241.3426514, 202.7079010, -242.1613312, 203.4472351, -444.7898254, 444.8691711
8: -290.9705505, 198.4971466, -291.7500916, 199.0496979, -490.0202637, 490.2472534
9: -219.5425873, 216.2140350, -220.4269104, 216.9405365, -436.4831238, 436.6409302

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2042231, upper bound: 461.2051102
time: 10.19 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2200899, upper bound: 461.2225511
time: 8.53 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -244.5776367, 193.4732056, -242.0842743, 191.5462341, -436.1238403, 435.5574951
1: -204.4175262, 171.7587585, -202.2670288, 170.0345459, -374.4520874, 374.0257874
2: -268.6275330, 174.2521973, -265.9084778, 172.5523376, -441.1798096, 440.1606750
3: -285.7423706, 150.8209229, -282.9065552, 149.2875214, -435.0299072, 433.7274475
4: -262.0225525, 200.7685699, -259.4692688, 198.7713928, -460.7939453, 460.2378540
5: -234.9153442, 182.4036560, -232.5094452, 180.7077179, -415.6230469, 414.9130859
6: -224.7715912, 215.6907806, -222.5233765, 213.5542908, -438.3258362, 438.2141724
7: -244.5365143, 205.3792267, -242.1613312, 203.4472351, -447.9837341, 447.5405273
8: -294.7610779, 201.0937653, -291.7500916, 199.0496979, -493.8107910, 492.8438110
9: -222.4592285, 219.0498047, -220.4269104, 216.9405365, -439.3997803, 439.4766541

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2057697, upper bound: 461.2056294
time: 8.80 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2213190, upper bound: 461.2229692
time: 9.86 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -243.5448151, 192.6747894, -242.0842743, 191.5462341, -435.0909729, 434.7590637
1: -203.5896149, 171.0831757, -202.2670288, 170.0345459, -373.6241455, 373.3502197
2: -267.5437622, 173.5565033, -265.9084778, 172.5523376, -440.0960388, 439.4649658
3: -284.6197510, 150.2388916, -282.9065552, 149.2875214, -433.9072876, 433.1454468
4: -260.9874573, 200.0178375, -259.4692688, 198.7713928, -459.7588501, 459.4871216
5: -233.9668427, 181.7662506, -232.5094452, 180.7077179, -414.6745605, 414.2756653
6: -223.8668671, 214.8315582, -222.5233765, 213.5542908, -437.4211426, 437.3549194
7: -243.5734253, 204.6076660, -242.1613312, 203.4472351, -447.0206299, 446.7689819
8: -293.5816956, 200.2887726, -291.7500916, 199.0496979, -492.6314087, 492.0388794
9: -221.6605225, 218.2372437, -220.4269104, 216.9405365, -438.6010742, 438.6640930

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2049695, upper bound: 461.2084952
time: 10.47 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251910, upper bound: 461.2256915
time: 8.32 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -246.2702332, 194.8287354, -242.0842743, 191.5462341, -437.8164673, 436.9129944
1: -205.8732300, 173.0046844, -202.2670288, 170.0345459, -375.9077759, 375.2717285
2: -270.5419922, 175.5079803, -265.9084778, 172.5523376, -443.0942688, 441.4164124
3: -287.8132019, 151.9130249, -282.9065552, 149.2875214, -437.1007080, 434.8195496
4: -263.9453735, 202.2661743, -259.4692688, 198.7713928, -462.7167664, 461.7354431
5: -236.5937195, 183.8229675, -232.5094452, 180.7077179, -417.3014526, 416.3323364
6: -226.3674469, 217.2335510, -222.5233765, 213.5542908, -439.9217224, 439.7569275
7: -246.3242340, 206.9079742, -242.1613312, 203.4472351, -449.7713928, 449.0691833
8: -296.8351135, 202.5220795, -291.7500916, 199.0496979, -495.8848267, 494.2721252
9: -224.1749573, 220.6789398, -220.4269104, 216.9405365, -441.1154785, 441.1058350

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2049695, upper bound: 461.2114178
time: 11.74 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259905
time: 7.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.81 seconds
IS_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2107724, upper bound: 461.2123965
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2206654, upper bound: 461.2234293
IS_A2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2121159, upper bound: 461.2128675
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2218927, upper bound: 461.2238378
IS_A2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2055152, upper bound: 461.2092504
IS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2257208, upper bound: 461.2265453
IS_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2098006, upper bound: 461.2121421
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2265406, upper bound: 461.2268540
IS_A2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2042231, upper bound: 461.2051102
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2200899, upper bound: 461.2225511
IS_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2057697, upper bound: 461.2056294
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2213190, upper bound: 461.2229692
IS_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2049695, upper bound: 461.2084952
IS_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2251910, upper bound: 461.2256915
IS_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2049695, upper bound: 461.2114178
IS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 20.81
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259905

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -237.4725342, 187.8997192, -429.3125610, 428.4440308
1: -201.7650757, 169.5303040, -198.4496002, 166.8235626, -368.5886230, 367.9798889
2: -265.1434326, 171.9864349, -260.8528137, 169.2942047, -434.4375916, 432.8392334
3: -282.0320129, 148.8755341, -277.5346985, 146.5005646, -428.5325928, 426.4102173
4: -258.5920715, 198.1590881, -254.5586090, 195.0419006, -453.6339722, 452.7176514
5: -231.8648071, 180.0232697, -228.1127167, 177.3268280, -409.1916199, 408.1359558
6: -221.8649445, 212.8996429, -218.2925568, 209.4876556, -431.3526001, 431.1921997
7: -241.3426514, 202.7079010, -237.5603333, 199.6000061, -440.9426575, 440.2682495
8: -290.9705505, 198.4971466, -286.2345581, 195.3197479, -486.2902832, 484.7316895
9: -219.5425873, 216.2140350, -216.2684631, 212.8459625, -432.3885498, 432.4824829

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2166251, upper bound: 461.2206653
time: 9.88 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1985963, upper bound: 461.1999180
time: 27.98 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -244.5776367, 193.4732056, -237.4725342, 187.8997192, -432.4773560, 430.9457397
1: -204.4175262, 171.7587585, -198.4496002, 166.8235626, -371.2410889, 370.2083435
2: -268.6275330, 174.2521973, -260.8528137, 169.2942047, -437.9216919, 435.1050110
3: -285.7423706, 150.8209229, -277.5346985, 146.5005646, -432.2429199, 428.3556213
4: -262.0225525, 200.7685699, -254.5586090, 195.0419006, -457.0644531, 455.3271484
5: -234.9153442, 182.4036560, -228.1127167, 177.3268280, -412.2421265, 410.5163269
6: -224.7715912, 215.6907806, -218.2925568, 209.4876556, -434.2592163, 433.9833374
7: -244.5365143, 205.3792267, -237.5603333, 199.6000061, -444.1365356, 442.9395752
8: -294.7610779, 201.0937653, -286.2345581, 195.3197479, -490.0808105, 487.3282776
9: -222.4592285, 219.0498047, -216.2684631, 212.8459625, -435.3051453, 435.3182373

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2197801, upper bound: 461.2215521
time: 10.29 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2164591, upper bound: 461.2195537
time: 7.91 seconds

## BFS IS instance: IS_A2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -241.9535065, 191.4205322, -239.0743866, 189.1624146, -431.1159058, 430.4949341
1: -202.2558746, 169.9739532, -199.7922211, 167.9400635, -370.1959229, 369.7661743
2: -265.8024597, 172.4334106, -262.6058350, 170.4243469, -436.2267761, 435.0392151
3: -282.7852478, 149.2650299, -279.3811340, 147.4806366, -430.2658691, 428.6461487
4: -259.2937927, 198.7212372, -256.2637939, 196.3474579, -455.6412354, 454.9850464
5: -232.4432526, 180.5945129, -229.6463013, 178.5066833, -410.9498596, 410.2408142
6: -222.4099579, 213.4340668, -219.7590637, 210.8946838, -433.3045959, 433.1930847
7: -241.9929962, 203.2898865, -239.1515045, 200.9265594, -442.9195557, 442.4414062
8: -291.6629028, 198.9813843, -288.1658630, 196.6365509, -488.2994385, 487.1472168
9: -220.2346954, 216.8199310, -217.7042542, 214.2737274, -434.5084229, 434.5241699

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2203326, upper bound: 461.2200667
time: 10.35 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2124111, upper bound: 461.2145129
time: 9.38 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -244.6727448, 193.5696716, -239.0743866, 189.1624146, -433.8351440, 432.6440430
1: -204.5349731, 171.8914642, -199.7922211, 167.9400635, -372.4750366, 371.6836853
2: -268.7944031, 174.3806152, -262.6058350, 170.4243469, -439.2187195, 436.9863892
3: -285.9725037, 150.9357147, -279.3811340, 147.4806366, -433.4531250, 430.3168030
4: -262.2454224, 200.9650879, -256.2637939, 196.3474579, -458.5928955, 457.2288818
5: -235.0643616, 182.6466064, -229.6463013, 178.5066833, -413.5709534, 412.2929077
6: -224.9051208, 215.8306427, -219.7590637, 210.8946838, -435.7997742, 435.5897217
7: -244.7378387, 205.5853424, -239.1515045, 200.9265594, -445.6643982, 444.7368469
8: -294.9093018, 201.2092896, -288.1658630, 196.6365509, -491.5458374, 489.3751526
9: -222.7432251, 219.2558746, -217.7042542, 214.2737274, -437.0169373, 436.9601135

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211636, upper bound: 461.2203003
time: 8.51 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2146778, upper bound: 461.2155709
time: 7.76 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -240.5336761, 190.3244171, -431.7372437, 431.5051880
1: -201.7650757, 169.5303040, -200.9682007, 168.9540253, -370.7191162, 370.4984741
2: -265.1434326, 171.9864349, -264.2121277, 171.4580841, -436.6015015, 436.1985474
3: -282.0320129, 148.8755341, -281.1194153, 148.3377075, -430.3697205, 429.9949341
4: -258.5920715, 198.1590881, -257.8196716, 197.5087585, -456.1008301, 455.9787292
5: -231.8648071, 180.0232697, -231.0254364, 179.5666351, -411.4314270, 411.0487061
6: -221.8649445, 212.8996429, -221.1036682, 212.1925201, -434.0574646, 434.0032654
7: -241.3426514, 202.7079010, -240.6218872, 202.1633911, -443.5059814, 443.3297729
8: -290.9705505, 198.4971466, -289.8797302, 197.7758026, -488.7463379, 488.3768921
9: -219.5425873, 216.2140350, -219.0381470, 215.5592651, -435.1018677, 435.2521973

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2048648, upper bound: 461.2110151
time: 10.77 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2003672, upper bound: 461.2048819
time: 10.01 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -244.5776367, 193.4732056, -240.5336761, 190.3244171, -434.9020386, 434.0068970
1: -204.4175262, 171.7587585, -200.9682007, 168.9540253, -373.3715515, 372.7269287
2: -268.6275330, 174.2521973, -264.2121277, 171.4580841, -440.0856018, 438.4643250
3: -285.7423706, 150.8209229, -281.1194153, 148.3377075, -434.0800781, 431.9403076
4: -262.0225525, 200.7685699, -257.8196716, 197.5087585, -459.5313110, 458.5882263
5: -234.9153442, 182.4036560, -231.0254364, 179.5666351, -414.4819641, 413.4290771
6: -224.7715912, 215.6907806, -221.1036682, 212.1925201, -436.9640503, 436.7944336
7: -244.5365143, 205.3792267, -240.6218872, 202.1633911, -446.6998901, 446.0010986
8: -294.7610779, 201.0937653, -289.8797302, 197.7758026, -492.5368652, 490.9735107
9: -222.4592285, 219.0498047, -219.0381470, 215.5592651, -438.0184631, 438.0879211

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2139469, upper bound: 461.2137311
time: 9.60 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2039823, upper bound: 461.2067307
time: 8.81 seconds

## BFS IS instance: IS_A2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -241.9535065, 191.4205322, -242.0842743, 191.5462341, -433.4997559, 433.5048218
1: -202.2558746, 169.9739532, -202.2670288, 170.0345459, -372.2904053, 372.2409668
2: -265.8024597, 172.4334106, -265.9084778, 172.5523376, -438.3547363, 438.3418579
3: -282.7852478, 149.2650299, -282.9065552, 149.2875214, -432.0727539, 432.1715698
4: -259.2937927, 198.7212372, -259.4692688, 198.7713928, -458.0651855, 458.1904907
5: -232.4432526, 180.5945129, -232.5094452, 180.7077179, -413.1509705, 413.1039124
6: -222.4099579, 213.4340668, -222.5233765, 213.5542908, -435.9642334, 435.9574585
7: -241.9929962, 203.2898865, -242.1613312, 203.4472351, -445.4402161, 445.4512024
8: -291.6629028, 198.9813843, -291.7500916, 199.0496979, -490.7125854, 490.7314453
9: -220.2346954, 216.8199310, -220.4269104, 216.9405365, -437.1752319, 437.2468262

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2238213, upper bound: 461.2240590
time: 10.09 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2224041, upper bound: 461.2231844
time: 7.97 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -244.6727448, 193.5696716, -242.0842743, 191.5462341, -436.2189331, 435.6539307
1: -204.5349731, 171.8914642, -202.2670288, 170.0345459, -374.5695190, 374.1584778
2: -268.7944031, 174.3806152, -265.9084778, 172.5523376, -441.3466797, 440.2890320
3: -285.9725037, 150.9357147, -282.9065552, 149.2875214, -435.2600098, 433.8422546
4: -262.2454224, 200.9650879, -259.4692688, 198.7713928, -461.0168152, 460.4343567
5: -235.0643616, 182.6466064, -232.5094452, 180.7077179, -415.7720947, 415.1560059
6: -224.9051208, 215.8306427, -222.5233765, 213.5542908, -438.4594116, 438.3540039
7: -244.7378387, 205.5853424, -242.1613312, 203.4472351, -448.1850586, 447.7466431
8: -294.9093018, 201.2092896, -291.7500916, 199.0496979, -493.9589844, 492.9593811
9: -222.7432251, 219.2558746, -220.4269104, 216.9405365, -439.6837769, 439.6827393

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2246744, upper bound: 461.2242940
time: 9.15 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2233223, upper bound: 461.2233223
time: 8.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.40 seconds
IS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2166251, upper bound: 461.2206653
IS_A2_B1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.1985963, upper bound: 461.1999180
IS_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2197801, upper bound: 461.2215521
IS_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2164591, upper bound: 461.2195537
IS_A2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2203326, upper bound: 461.2200667
IS_A2_B1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2124111, upper bound: 461.2145129
IS_A2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2211636, upper bound: 461.2203003
IS_A2_B1_A2_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2146778, upper bound: 461.2155709
IS_A2_B2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2048648, upper bound: 461.2110151
IS_A2_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2003672, upper bound: 461.2048819
IS_A2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2139469, upper bound: 461.2137311
IS_A2_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2039823, upper bound: 461.2067307
IS_A2_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2238213, upper bound: 461.2240590
IS_A2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2224041, upper bound: 461.2231844
IS_A2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2246744, upper bound: 461.2242940
IS_A2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 7, lower bound: -461.2233223, upper bound: 461.2233223

## BFS IS instance: IS_A2_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -232.4122009, 183.9010315, -425.3139343, 423.3837280
1: -201.7650757, 169.5303040, -194.2057800, 163.2667542, -365.0317993, 363.7360535
2: -265.1434326, 171.9864349, -255.2798920, 165.7097321, -430.8531494, 427.2663269
3: -282.0320129, 148.8755341, -271.6424255, 143.3794250, -425.4114380, 420.5179443
4: -258.5920715, 198.1590881, -249.1179504, 190.8704529, -449.4625244, 447.2769775
5: -231.8648071, 180.0232697, -223.2478180, 173.5440521, -405.4088135, 403.2710876
6: -221.8649445, 212.8996429, -213.6491394, 205.0198364, -426.8847656, 426.5487366
7: -241.3426514, 202.7079010, -232.4963684, 195.3652649, -436.7079163, 435.2042847
8: -290.9705505, 198.4971466, -280.1150513, 191.1208038, -482.0913696, 478.6121826
9: -219.5425873, 216.2140350, -211.6643982, 208.3007507, -427.8433228, 427.8784180

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2106529, upper bound: 461.2130587
time: 10.83 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1951026, upper bound: 461.2010571
time: 10.01 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 22.14 seconds
IS_A2_B1_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 22.14
Output dim: 7, lower bound: -461.2106529, upper bound: 461.2130587
IS_A2_B1_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 22.14
Output dim: 7, lower bound: -461.1951026, upper bound: 461.2010571
IS_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2197801, upper bound: 461.2215521
IS_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2164591, upper bound: 461.2195537
IS_A2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2203326, upper bound: 461.2200667
IS_A2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2211636, upper bound: 461.2203003
IS_A2_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2238213, upper bound: 461.2240590
IS_A2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2224041, upper bound: 461.2231844
IS_A2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2246744, upper bound: 461.2242940
IS_A2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 22.14
Output dim: 7, lower bound: -461.2233223, upper bound: 461.2233223
Binary search (step 2): status=Status.UNKNOWN, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=464.3514404296875
rel_dist={7: [-461.2313740276876, 461.23137391660043]}

## Binary search (step 3) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2143547, upper bound: 461.2194728
time: 9.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2313655, upper bound: 461.2313653
time: 7.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.00 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 17.00
Output dim: 7, lower bound: -461.2143547, upper bound: 461.2194728
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.00
Output dim: 7, lower bound: -461.2313655, upper bound: 461.2313653

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -252.2428284, 199.5615692, -451.7855835, 451.7895203
1: -210.8763428, 177.2277527, -210.8920135, 177.2409058, -388.1171875, 388.1196899
2: -277.1233826, 179.7556458, -277.1440125, 179.7689209, -456.8922729, 456.8995667
3: -294.8307190, 155.5760193, -294.8527222, 155.5875702, -450.4182739, 450.4287109
4: -270.4299011, 207.2068634, -270.4500427, 207.2221527, -477.6520386, 477.6569214
5: -242.3272858, 188.3644562, -242.3452606, 188.3783875, -430.7056580, 430.7096863
6: -231.8549500, 222.5227814, -231.8721466, 222.5393829, -454.3942871, 454.3949280
7: -252.3645477, 211.9523926, -252.3833771, 211.9680939, -464.3326111, 464.3357544
8: -304.0132751, 207.4294128, -304.0357361, 207.4447021, -511.4579773, 511.4651489
9: -229.6954498, 226.0778656, -229.7125092, 226.0945892, -455.7900391, 455.7903442

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2276531, upper bound: 461.2279180
time: 8.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271467, upper bound: 461.2271468
time: 7.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.04 seconds
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.04
Output dim: 7, lower bound: -461.2276531, upper bound: 461.2279180
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.04
Output dim: 7, lower bound: -461.2271467, upper bound: 461.2271468

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -239.0743866, 189.1624146, -441.3864441, 438.6211243
1: -210.8763428, 177.2277527, -199.7922211, 167.9400635, -378.8164062, 377.0199280
2: -277.1233826, 179.7556458, -262.6058350, 170.4243469, -447.5476990, 442.3613892
3: -294.8307190, 155.5760193, -279.3811340, 147.4806366, -442.3113403, 434.9571228
4: -270.4299011, 207.2068634, -256.2637939, 196.3474579, -466.7773438, 463.4706421
5: -242.3272858, 188.3644562, -229.6463013, 178.5066833, -420.8338623, 418.0107422
6: -231.8549500, 222.5227814, -219.7590637, 210.8946838, -442.7495728, 442.2818298
7: -252.3645477, 211.9523926, -239.1515045, 200.9265594, -453.2911072, 451.1038818
8: -304.0132751, 207.4294128, -288.1658630, 196.6365509, -500.6498413, 495.5952759
9: -229.6954498, 226.0778656, -217.7042542, 214.2737274, -443.9691467, 443.7821045

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2236378, upper bound: 461.2250843
time: 9.65 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2276425, upper bound: 461.2278818
time: 8.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -252.2240906, 199.5467682, -242.0842743, 191.5462341, -443.7702332, 441.6310120
1: -210.8763428, 177.2277527, -202.2670288, 170.0345459, -380.9108887, 379.4947510
2: -277.1233826, 179.7556458, -265.9084778, 172.5523376, -449.6757202, 445.6640320
3: -294.8307190, 155.5760193, -282.9065552, 149.2875214, -444.1182251, 438.4825439
4: -270.4299011, 207.2068634, -259.4692688, 198.7713928, -469.2012939, 466.6761475
5: -242.3272858, 188.3644562, -232.5094452, 180.7077179, -423.0350037, 420.8738708
6: -231.8549500, 222.5227814, -222.5233765, 213.5542908, -445.4092102, 445.0461426
7: -252.3645477, 211.9523926, -242.1613312, 203.4472351, -455.8117065, 454.1136780
8: -304.0132751, 207.4294128, -291.7500916, 199.0496979, -503.0629883, 499.1795044
9: -229.6954498, 226.0778656, -220.4269104, 216.9405365, -446.6359863, 446.5047607

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2231367, upper bound: 461.2243538
time: 10.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271391, upper bound: 461.2271391
time: 8.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.27 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.27
Output dim: 7, lower bound: -461.2236378, upper bound: 461.2250843
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.27
Output dim: 7, lower bound: -461.2276425, upper bound: 461.2278818
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.27
Output dim: 7, lower bound: -461.2231367, upper bound: 461.2243538
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.27
Output dim: 7, lower bound: -461.2271391, upper bound: 461.2271391

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -248.3659973, 196.4675598, -239.0339050, 189.1302338, -437.4962158, 435.5014648
1: -207.5953979, 174.4416504, -199.7581787, 167.9113922, -375.5067749, 374.1998291
2: -272.8066406, 176.9463654, -262.5610046, 170.3953552, -443.2019958, 439.5072937
3: -290.1911926, 153.1420135, -279.3332520, 147.4555817, -437.6467896, 432.4752808
4: -266.1442566, 203.8983154, -256.2197571, 196.3137817, -462.4580383, 460.1180725
5: -238.5536957, 185.2788849, -229.6072235, 178.4757843, -417.0294800, 414.8861084
6: -228.2519379, 219.0505829, -219.7215729, 210.8586884, -439.1106262, 438.7721558
7: -248.3798370, 208.5741119, -239.1105347, 200.8920898, -449.2718506, 447.6846313
8: -299.3194885, 204.2174377, -288.1169128, 196.6031952, -495.9226685, 492.3343506
9: -225.9566650, 222.4711304, -217.6665649, 214.2368622, -440.1935425, 440.1376953

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2125903, upper bound: 461.2131877
time: 7.97 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2236326, upper bound: 461.2250843
time: 9.71 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -250.1731720, 197.9147339, -239.0743866, 189.1624146, -439.3355408, 436.9891357
1: -209.1479492, 175.7681427, -199.7922211, 167.9400635, -377.0880127, 375.5603638
2: -274.8470154, 178.2832031, -262.6058350, 170.4243469, -445.2713318, 440.8890076
3: -292.3962097, 154.3041382, -279.3811340, 147.4806366, -439.8768311, 433.6852417
4: -268.1894531, 205.4903107, -256.2637939, 196.3474579, -464.5369263, 461.7540894
5: -240.3438568, 186.7873077, -229.6463013, 178.5066833, -418.8504944, 416.4335938
6: -229.9516754, 220.6946259, -219.7590637, 210.8946838, -440.8463440, 440.4536438
7: -250.2817993, 210.1995087, -239.1515045, 200.9265594, -451.2083740, 449.3510132
8: -301.5317688, 205.7399750, -288.1658630, 196.6365509, -498.1683350, 493.9058228
9: -227.7773590, 224.2036438, -217.7042542, 214.2737274, -442.0510864, 441.9078674

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2107203, upper bound: 461.2123313
time: 9.12 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2276337, upper bound: 461.2278680
time: 9.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -248.3659973, 196.4675598, -242.0431366, 191.5135498, -439.8795471, 438.5106812
1: -207.5953979, 174.4416504, -202.2324371, 170.0054016, -377.6007996, 376.6740723
2: -272.8066406, 176.9463654, -265.8629761, 172.5228882, -445.3294983, 442.8092651
3: -290.1911926, 153.1420135, -282.8578796, 149.2620697, -439.4532471, 435.9998779
4: -266.1442566, 203.8983154, -259.4245911, 198.7371674, -464.8814087, 463.3229065
5: -238.5536957, 185.2788849, -232.4697571, 180.6763611, -419.2300415, 417.7486267
6: -228.2519379, 219.0505829, -222.4853516, 213.5177155, -441.7695923, 441.5359497
7: -248.3798370, 208.5741119, -242.1197510, 203.4122467, -451.7919922, 450.6938171
8: -299.3194885, 204.2174377, -291.7003479, 199.0158081, -498.3352966, 495.9177856
9: -225.9566650, 222.4711304, -220.3886566, 216.9030914, -442.8597412, 442.8598022

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2199058, upper bound: 461.2222892
time: 9.38 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211267, upper bound: 461.2227522
time: 9.15 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -250.1731720, 197.9147339, -242.0842743, 191.5462341, -441.7193298, 439.9990234
1: -209.1479492, 175.7681427, -202.2670288, 170.0345459, -379.1824951, 378.0351562
2: -274.8470154, 178.2832031, -265.9084778, 172.5523376, -447.3993225, 444.1916504
3: -292.3962097, 154.3041382, -282.9065552, 149.2875214, -441.6837158, 437.2106934
4: -268.1894531, 205.4903107, -259.4692688, 198.7713928, -466.9608459, 464.9595947
5: -240.3438568, 186.7873077, -232.5094452, 180.7077179, -421.0515747, 419.2967529
6: -229.9516754, 220.6946259, -222.5233765, 213.5542908, -443.5059509, 443.2180176
7: -250.2817993, 210.1995087, -242.1613312, 203.4472351, -453.7290344, 452.3607788
8: -301.5317688, 205.7399750, -291.7500916, 199.0496979, -500.5814819, 497.4900513
9: -227.7773590, 224.2036438, -220.4269104, 216.9405365, -444.7178955, 444.6305237

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2102457, upper bound: 461.2117014
time: 9.63 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271249, upper bound: 461.2271249
time: 10.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.67 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2125903, upper bound: 461.2131877
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2236326, upper bound: 461.2250843
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2107203, upper bound: 461.2123313
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2276337, upper bound: 461.2278680
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2199058, upper bound: 461.2222892
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2211267, upper bound: 461.2227522
IS_A2_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2102457, upper bound: 461.2117014
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.67
Output dim: 7, lower bound: -461.2271249, upper bound: 461.2271249

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -248.3659973, 196.4675598, -237.4320221, 187.8675079, -436.2335205, 433.8995972
1: -207.5953979, 174.4416504, -198.4155273, 166.7948456, -374.3901672, 372.8571777
2: -272.8066406, 176.9463654, -260.8079224, 169.2652130, -442.0718384, 437.7541809
3: -290.1911926, 153.1420135, -277.4867249, 146.4754791, -436.6666565, 430.6287231
4: -266.1442566, 203.8983154, -254.5145416, 195.0082092, -461.1524658, 458.4128418
5: -238.5536957, 185.2788849, -228.0735779, 177.2959290, -415.8496094, 413.3524170
6: -228.2519379, 219.0505829, -218.2550659, 209.4516296, -437.7035522, 437.3056641
7: -248.3798370, 208.5741119, -237.5193329, 199.5655060, -447.9452820, 446.0934143
8: -299.3194885, 204.2174377, -286.1856384, 195.2863922, -494.6058655, 490.4030762
9: -225.9566650, 222.4711304, -216.2307281, 212.8090668, -438.7657166, 438.7018433

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2204713, upper bound: 461.2231309
time: 8.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2216848, upper bound: 461.2235747
time: 11.73 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -248.5842285, 196.6623840, -239.0743866, 189.1624146, -437.7466431, 435.7367554
1: -207.8164368, 174.6607208, -199.7922211, 167.9400635, -375.7564697, 374.4529419
2: -273.1082458, 177.1617584, -262.6058350, 170.4243469, -443.5325928, 439.7675171
3: -290.5650024, 153.3318787, -279.3811340, 147.4806366, -438.0456543, 432.7130127
4: -266.4980469, 204.1960297, -256.2637939, 196.3474579, -462.8455200, 460.4598389
5: -238.8228455, 185.6174316, -229.6463013, 178.5066833, -417.3294067, 415.2637329
6: -228.4969788, 219.2989960, -219.7590637, 210.8946838, -439.3916321, 439.0580444
7: -248.7038116, 208.8838806, -239.1515045, 200.9265594, -449.6303711, 448.0353699
8: -299.6156311, 204.4341583, -288.1658630, 196.6365509, -496.2521973, 492.6000061
9: -226.3535767, 222.7880402, -217.7042542, 214.2737274, -440.6273193, 440.4923096

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2256607, upper bound: 461.2264815
time: 9.14 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2265014, upper bound: 461.2268031
time: 9.02 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -242.0431366, 191.5135498, -432.9263916, 433.0146484
1: -201.7650757, 169.5303040, -202.2324371, 170.0054016, -371.7704773, 371.7627258
2: -265.1434326, 171.9864349, -265.8629761, 172.5228882, -437.6662903, 437.8494263
3: -282.0320129, 148.8755341, -282.8578796, 149.2620697, -431.2940674, 431.7333984
4: -258.5920715, 198.1590881, -259.4245911, 198.7371674, -457.3292236, 457.5836792
5: -231.8648071, 180.0232697, -232.4697571, 180.6763611, -412.5411682, 412.4930420
6: -221.8649445, 212.8996429, -222.4853516, 213.5177155, -435.3826599, 435.3850098
7: -241.3426514, 202.7079010, -242.1197510, 203.4122467, -444.7548218, 444.8276367
8: -290.9705505, 198.4971466, -291.7003479, 199.0158081, -489.9863586, 490.1975098
9: -219.5425873, 216.2140350, -220.3886566, 216.9030914, -436.4456787, 436.6026917

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2020339, upper bound: 461.2032012
time: 10.88 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2199058, upper bound: 461.2222892
time: 10.13 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -244.5776367, 193.4732056, -242.0431366, 191.5135498, -436.0911865, 435.5163574
1: -204.4175262, 171.7587585, -202.2324371, 170.0054016, -374.4229126, 373.9911804
2: -268.6275330, 174.2521973, -265.8629761, 172.5228882, -441.1503906, 440.1151733
3: -285.7423706, 150.8209229, -282.8578796, 149.2620697, -435.0044250, 433.6787415
4: -262.0225525, 200.7685699, -259.4245911, 198.7371674, -460.7597046, 460.1931763
5: -234.9153442, 182.4036560, -232.4697571, 180.6763611, -415.5916748, 414.8734131
6: -224.7715912, 215.6907806, -222.4853516, 213.5177155, -438.2892151, 438.1761475
7: -244.5365143, 205.3792267, -242.1197510, 203.4122467, -447.9487000, 447.4989624
8: -294.7610779, 201.0937653, -291.7003479, 199.0158081, -493.7768860, 492.7940979
9: -222.4592285, 219.0498047, -220.3886566, 216.9030914, -439.3622742, 439.4384155

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2039695, upper bound: 461.2040202
time: 9.63 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2211267, upper bound: 461.2227522
time: 9.78 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -248.5842285, 196.6623840, -242.0842743, 191.5462341, -440.1304626, 438.7466431
1: -207.8164368, 174.6607208, -202.2670288, 170.0345459, -377.8509827, 376.9277344
2: -273.1082458, 177.1617584, -265.9084778, 172.5523376, -445.6605530, 443.0701599
3: -290.5650024, 153.3318787, -282.9065552, 149.2875214, -439.8525391, 436.2384338
4: -266.4980469, 204.1960297, -259.4692688, 198.7713928, -465.2694397, 463.6652832
5: -238.8228455, 185.6174316, -232.5094452, 180.7077179, -419.5305481, 418.1268616
6: -228.4969788, 219.2989960, -222.5233765, 213.5542908, -442.0512390, 441.8223877
7: -248.7038116, 208.8838806, -242.1613312, 203.4472351, -452.1510315, 451.0451355
8: -299.6156311, 204.4341583, -291.7500916, 199.0496979, -498.6653442, 496.1842651
9: -226.3535767, 222.7880402, -220.4269104, 216.9405365, -443.2941284, 443.2149658

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251595, upper bound: 461.2256605
time: 8.51 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259727, upper bound: 461.2259727
time: 8.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.06 seconds
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2204713, upper bound: 461.2231309
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2216848, upper bound: 461.2235747
IS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2256607, upper bound: 461.2264815
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2265014, upper bound: 461.2268031
IS_A2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2020339, upper bound: 461.2032012
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2199058, upper bound: 461.2222892
IS_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2039695, upper bound: 461.2040202
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2211267, upper bound: 461.2227522
IS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2251595, upper bound: 461.2256605
IS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 7, lower bound: -461.2259727, upper bound: 461.2259727

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -237.4320221, 187.8675079, -429.2803955, 428.4035339
1: -201.7650757, 169.5303040, -198.4155273, 166.7948456, -368.5599060, 367.9458008
2: -265.1434326, 171.9864349, -260.8079224, 169.2652130, -434.4086304, 432.7943726
3: -282.0320129, 148.8755341, -277.4867249, 146.4754791, -428.5074768, 426.3622437
4: -258.5920715, 198.1590881, -254.5145416, 195.0082092, -453.6002502, 452.6736145
5: -231.8648071, 180.0232697, -228.0735779, 177.2959290, -409.1607056, 408.0968628
6: -221.8649445, 212.8996429, -218.2550659, 209.4516296, -431.3165894, 431.1547241
7: -241.3426514, 202.7079010, -237.5193329, 199.5655060, -440.9081116, 440.2272034
8: -290.9705505, 198.4971466, -286.1856384, 195.2863922, -486.2569580, 484.6827698
9: -219.5425873, 216.2140350, -216.2307281, 212.8090668, -432.3516235, 432.4447632

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2163013, upper bound: 461.2201387
time: 9.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2152836, upper bound: 461.2187802
time: 8.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -244.5776367, 193.4732056, -237.4320221, 187.8675079, -432.4451294, 430.9052124
1: -204.4175262, 171.7587585, -198.4155273, 166.7948456, -371.2123413, 370.1742249
2: -268.6275330, 174.2521973, -260.8079224, 169.2652130, -437.8927307, 435.0601196
3: -285.7423706, 150.8209229, -277.4867249, 146.4754791, -432.2178345, 428.3076172
4: -262.0225525, 200.7685699, -254.5145416, 195.0082092, -457.0307617, 455.2830811
5: -234.9153442, 182.4036560, -228.0735779, 177.2959290, -412.2112122, 410.4771729
6: -224.7715912, 215.6907806, -218.2550659, 209.4516296, -434.2231750, 433.9458618
7: -244.5365143, 205.3792267, -237.5193329, 199.5655060, -444.1019897, 442.8985596
8: -294.7610779, 201.0937653, -286.1856384, 195.2863922, -490.0474854, 487.2793274
9: -222.4592285, 219.0498047, -216.2307281, 212.8090668, -435.2682190, 435.2805176

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2194730, upper bound: 461.2211984
time: 10.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2162352, upper bound: 461.2191870
time: 12.84 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -241.9535065, 191.4205322, -239.0743866, 189.1624146, -431.1159058, 430.4949341
1: -202.2558746, 169.9739532, -199.7922211, 167.9400635, -370.1959229, 369.7661743
2: -265.8024597, 172.4334106, -262.6058350, 170.4243469, -436.2267761, 435.0392151
3: -282.7852478, 149.2650299, -279.3811340, 147.4806366, -430.2658691, 428.6461487
4: -259.2937927, 198.7212372, -256.2637939, 196.3474579, -455.6412354, 454.9850464
5: -232.4432526, 180.5945129, -229.6463013, 178.5066833, -410.9498596, 410.2408142
6: -222.4099579, 213.4340668, -219.7590637, 210.8946838, -433.3045959, 433.1930847
7: -241.9929962, 203.2898865, -239.1515045, 200.9265594, -442.9195557, 442.4414062
8: -291.6629028, 198.9813843, -288.1658630, 196.6365509, -488.2994385, 487.1472168
9: -220.2346954, 216.8199310, -217.7042542, 214.2737274, -434.5084229, 434.5241699

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2199531, upper bound: 461.2197719
time: 10.42 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2122269, upper bound: 461.2142605
time: 8.42 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -244.6727448, 193.5696716, -239.0743866, 189.1624146, -433.8351440, 432.6440430
1: -204.5349731, 171.8914642, -199.7922211, 167.9400635, -372.4750366, 371.6836853
2: -268.7944031, 174.3806152, -262.6058350, 170.4243469, -439.2187195, 436.9863892
3: -285.9725037, 150.9357147, -279.3811340, 147.4806366, -433.4531250, 430.3168030
4: -262.2454224, 200.9650879, -256.2637939, 196.3474579, -458.5928955, 457.2288818
5: -235.0643616, 182.6466064, -229.6463013, 178.5066833, -413.5709534, 412.2929077
6: -224.9051208, 215.8306427, -219.7590637, 210.8946838, -435.7997742, 435.5897217
7: -244.7378387, 205.5853424, -239.1515045, 200.9265594, -445.6643982, 444.7368469
8: -294.9093018, 201.2092896, -288.1658630, 196.6365509, -491.5458374, 489.3751526
9: -222.7432251, 219.2558746, -217.7042542, 214.2737274, -437.0169373, 436.9601135

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2208222, upper bound: 461.2200169
time: 10.42 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2145280, upper bound: 461.2154271
time: 7.81 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -240.4925690, 190.2917023, -431.7045593, 431.4640808
1: -201.7650757, 169.5303040, -200.9335632, 168.9248657, -370.6899414, 370.4638062
2: -265.1434326, 171.9864349, -264.1665955, 171.4286194, -436.5720215, 436.1530151
3: -282.0320129, 148.8755341, -281.0707092, 148.3122559, -430.3442688, 429.9462280
4: -258.5920715, 198.1590881, -257.7748718, 197.4745331, -456.0665894, 455.9339600
5: -231.8648071, 180.0232697, -230.9857330, 179.5352325, -411.3999939, 411.0090027
6: -221.8649445, 212.8996429, -221.0655823, 212.1559601, -434.0209045, 433.9652100
7: -241.3426514, 202.7079010, -240.5803070, 202.1283722, -443.4710083, 443.2882080
8: -290.9705505, 198.4971466, -289.8300476, 197.7419128, -488.7124634, 488.3272095
9: -219.5425873, 216.2140350, -218.9998627, 215.5218048, -435.0643616, 435.2138977

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2042884, upper bound: 461.2101223
time: 11.01 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1999672, upper bound: 461.2042650
time: 9.83 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -244.5776367, 193.4732056, -240.4925690, 190.2917023, -434.8693237, 433.9657593
1: -204.4175262, 171.7587585, -200.9335632, 168.9248657, -373.3423767, 372.6922302
2: -268.6275330, 174.2521973, -264.1665955, 171.4286194, -440.0561523, 438.4187927
3: -285.7423706, 150.8209229, -281.0707092, 148.3122559, -434.0546265, 431.8916321
4: -262.0225525, 200.7685699, -257.7748718, 197.4745331, -459.4970703, 458.5434570
5: -234.9153442, 182.4036560, -230.9857330, 179.5352325, -414.4505310, 413.3894043
6: -224.7715912, 215.6907806, -221.0655823, 212.1559601, -436.9275208, 436.7563477
7: -244.5365143, 205.3792267, -240.5803070, 202.1283722, -446.6648865, 445.9595337
8: -294.7610779, 201.0937653, -289.8300476, 197.7419128, -492.5029907, 490.9237671
9: -222.4592285, 219.0498047, -218.9998627, 215.5218048, -437.9809875, 438.0496216

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2132998, upper bound: 461.2130787
time: 10.55 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2036689, upper bound: 461.2062854
time: 10.66 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -241.9535065, 191.4205322, -242.0842743, 191.5462341, -433.4997559, 433.5048218
1: -202.2558746, 169.9739532, -202.2670288, 170.0345459, -372.2904053, 372.2409668
2: -265.8024597, 172.4334106, -265.9084778, 172.5523376, -438.3547363, 438.3418579
3: -282.7852478, 149.2650299, -282.9065552, 149.2875214, -432.0727539, 432.1715698
4: -259.2937927, 198.7212372, -259.4692688, 198.7713928, -458.0651855, 458.1904907
5: -232.4432526, 180.5945129, -232.5094452, 180.7077179, -413.1509705, 413.1039124
6: -222.4099579, 213.4340668, -222.5233765, 213.5542908, -435.9642334, 435.9574585
7: -241.9929962, 203.2898865, -242.1613312, 203.4472351, -445.4402161, 445.4512024
8: -291.6629028, 198.9813843, -291.7500916, 199.0496979, -490.7125854, 490.7314453
9: -220.2346954, 216.8199310, -220.4269104, 216.9405365, -437.1752319, 437.2468262

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2237387, upper bound: 461.2239631
time: 8.76 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2223725, upper bound: 461.2231336
time: 8.97 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -244.6727448, 193.5696716, -242.0842743, 191.5462341, -436.2189331, 435.6539307
1: -204.5349731, 171.8914642, -202.2670288, 170.0345459, -374.5695190, 374.1584778
2: -268.7944031, 174.3806152, -265.9084778, 172.5523376, -441.3466797, 440.2890320
3: -285.9725037, 150.9357147, -282.9065552, 149.2875214, -435.2600098, 433.8422546
4: -262.2454224, 200.9650879, -259.4692688, 198.7713928, -461.0168152, 460.4343567
5: -235.0643616, 182.6466064, -232.5094452, 180.7077179, -415.7720947, 415.1560059
6: -224.9051208, 215.8306427, -222.5233765, 213.5542908, -438.4594116, 438.3540039
7: -244.7378387, 205.5853424, -242.1613312, 203.4472351, -448.1850586, 447.7466431
8: -294.9093018, 201.2092896, -291.7500916, 199.0496979, -493.9589844, 492.9593811
9: -222.7432251, 219.2558746, -220.4269104, 216.9405365, -439.6837769, 439.6827393

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2245958, upper bound: 461.2242025
time: 9.61 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2232898, upper bound: 461.2232898
time: 9.28 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 20.15 seconds
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2163013, upper bound: 461.2201387
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2152836, upper bound: 461.2187802
IS_A2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2194730, upper bound: 461.2211984
IS_A2_B1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2162352, upper bound: 461.2191870
IS_A2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2199531, upper bound: 461.2197719
IS_A2_B1_A2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2122269, upper bound: 461.2142605
IS_A2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2208222, upper bound: 461.2200169
IS_A2_B1_A2_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2145280, upper bound: 461.2154271
IS_A2_B2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2042884, upper bound: 461.2101223
IS_A2_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.1999672, upper bound: 461.2042650
IS_A2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2132998, upper bound: 461.2130787
IS_A2_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2036689, upper bound: 461.2062854
IS_A2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2237387, upper bound: 461.2239631
IS_A2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2223725, upper bound: 461.2231336
IS_A2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2245958, upper bound: 461.2242025
IS_A2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 20.15
Output dim: 7, lower bound: -461.2232898, upper bound: 461.2232898

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -241.4129028, 190.9715118, -232.3724060, 183.8693848, -425.2822571, 423.3439331
1: -201.7650757, 169.5303040, -194.1722717, 163.2385406, -365.0036011, 363.7025146
2: -265.1434326, 171.9864349, -255.2358398, 165.6812134, -430.8246460, 427.2222900
3: -282.0320129, 148.8755341, -271.5953369, 143.3547821, -425.3867798, 420.4708862
4: -258.5920715, 198.1590881, -249.0746765, 190.8373718, -449.4293823, 447.2337341
5: -231.8648071, 180.0232697, -223.2093964, 173.5136566, -405.3784485, 403.2326660
6: -221.8649445, 212.8996429, -213.6123047, 204.9844513, -426.8493958, 426.5118408
7: -241.3426514, 202.7079010, -232.4561310, 195.3313751, -436.6740112, 435.1640320
8: -290.9705505, 198.4971466, -280.0668945, 191.0880127, -482.0585632, 478.5640259
9: -219.5425873, 216.2140350, -211.6273651, 208.2644958, -427.8070374, 427.8414001

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2098433, upper bound: 461.2119419
time: 9.26 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1944120, upper bound: 461.1998551
time: 10.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -239.4979706, 189.4588165, -237.4320221, 187.8675079, -427.3654785, 426.8908386
1: -200.1567535, 168.1884155, -198.4155273, 166.7948456, -366.9515991, 366.6039429
2: -263.0338745, 170.6532440, -260.8079224, 169.2652130, -432.2990723, 431.4611511
3: -279.8296509, 147.6879120, -277.4867249, 146.4754791, -426.3050842, 425.1746216
4: -256.5614624, 196.5801239, -254.5145416, 195.0082092, -451.5696716, 451.0946655
5: -230.0317383, 178.6060181, -228.0735779, 177.2959290, -407.3276062, 406.6795044
6: -220.1104584, 211.2065125, -218.2550659, 209.4516296, -429.5620728, 429.4615784
7: -239.4531097, 201.1283569, -237.5193329, 199.5655060, -439.0185547, 438.6476440
8: -288.6191406, 196.8774261, -286.1856384, 195.2863922, -483.9055176, 483.0630493
9: -217.8389893, 214.4870911, -216.2307281, 212.8090668, -430.6480408, 430.7178040

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2057504, upper bound: 461.2098997
time: 9.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2020776, upper bound: 461.2049011
time: 9.27 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -240.0702972, 189.9336700, -239.0743866, 189.1624146, -429.2327271, 429.0080261
1: -200.6823730, 168.6547699, -199.7922211, 167.9400635, -368.6224365, 368.4469910
2: -263.7349243, 171.0950012, -262.6058350, 170.4243469, -434.1592407, 433.7008362
3: -280.5787354, 148.1040497, -279.3811340, 147.4806366, -428.0593872, 427.4851074
4: -257.2617493, 197.1809692, -256.2637939, 196.3474579, -453.6091919, 453.4447632
5: -230.6348419, 179.1903076, -229.6463013, 178.5066833, -409.1413879, 408.8366089
6: -220.6755066, 211.7738953, -219.7590637, 210.8946838, -431.5701294, 431.5329590
7: -240.1009369, 201.7074280, -239.1515045, 200.9265594, -441.0274963, 440.8589478
8: -289.4053345, 197.4399567, -288.1658630, 196.6365509, -486.0418701, 485.6058350
9: -218.5163574, 215.1402130, -217.7042542, 214.2737274, -432.7901001, 432.8444824

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2177206, upper bound: 461.2180830
time: 9.15 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2161788, upper bound: 461.2158195
time: 10.66 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -242.8137665, 192.1016846, -239.0743866, 189.1624146, -431.9761353, 431.1760559
1: -202.9814301, 170.5889282, -199.7922211, 167.9400635, -370.9215088, 370.3811646
2: -266.7531128, 173.0595093, -262.6058350, 170.4243469, -437.1774292, 435.6653442
3: -283.7941589, 149.7894592, -279.3811340, 147.4806366, -431.2747803, 429.1705322
4: -260.2388916, 199.4443817, -256.2637939, 196.3474579, -456.5863647, 455.7081604
5: -233.2790527, 181.2599792, -229.6463013, 178.5066833, -411.7857056, 410.9062805
6: -223.1928101, 214.1915131, -219.7590637, 210.8946838, -434.0874329, 433.9505310
7: -242.8695831, 204.0229340, -239.1515045, 200.9265594, -443.7961426, 443.1744385
8: -292.6805725, 199.6871948, -288.1658630, 196.6365509, -489.3171387, 487.8530579
9: -221.0466156, 217.5976562, -217.7042542, 214.2737274, -435.3203430, 435.3018799

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2183982, upper bound: 461.2167593
time: 8.31 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2171483, upper bound: 461.2160555
time: 9.17 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -236.7963867, 187.3468628, -242.0842743, 191.5462341, -428.3426208, 429.4311218
1: -197.9313507, 166.3496552, -202.2670288, 170.0345459, -367.9658813, 368.6166992
2: -260.1238708, 168.7811737, -265.9084778, 172.5523376, -432.6761475, 434.6896362
3: -276.7794800, 146.0843353, -282.9065552, 149.2875214, -426.0670166, 428.9908752
4: -253.7505646, 194.4707489, -259.4692688, 198.7713928, -452.5219727, 453.9400024
5: -227.4861450, 176.7407990, -232.5094452, 180.7077179, -408.1938477, 409.2502441
6: -217.6777649, 208.8818817, -222.5233765, 213.5542908, -431.2320251, 431.4052734
7: -236.8318939, 198.9746857, -242.1613312, 203.4472351, -440.2791138, 441.1359558
8: -285.4294434, 194.7030792, -291.7500916, 199.0496979, -484.4791260, 486.4531250
9: -215.5433655, 212.1882172, -220.4269104, 216.9405365, -432.4838867, 432.6151123

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2156068, upper bound: 461.2139165
time: 9.60 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2077265, upper bound: 461.2086362
time: 10.59 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -241.2787018, 190.9252319, -242.0842743, 191.5462341, -432.8248596, 433.0095215
1: -201.6721649, 169.4802246, -202.2670288, 170.0345459, -371.7067261, 371.7472534
2: -265.0616150, 171.9299316, -265.9084778, 172.5523376, -437.6138916, 437.8383789
3: -282.1057129, 148.7448120, -282.9065552, 149.2875214, -431.3932495, 431.6513672
4: -258.5793457, 198.1199036, -259.4692688, 198.7713928, -457.3507080, 457.5891724
5: -231.8060760, 180.0293427, -232.5094452, 180.7077179, -412.5137939, 412.5387878
6: -221.8263550, 212.8628387, -222.5233765, 213.5542908, -435.3806152, 435.3862305
7: -241.3310547, 202.7260895, -242.1613312, 203.4472351, -444.7782288, 444.8873901
8: -290.7870483, 198.2613220, -291.7500916, 199.0496979, -489.8367310, 490.0114136
9: -219.6292572, 216.1915894, -220.4269104, 216.9405365, -436.5697937, 436.6184998

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2106504, upper bound: 461.2140837
time: 11.15 seconds

## Relational analysis of IS_A2_B2_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2060018, upper bound: 461.2075165
time: 10.04 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -239.5253906, 189.5019073, -242.0842743, 191.5462341, -431.0715637, 431.5861511
1: -200.2181854, 168.2736969, -202.2670288, 170.0345459, -370.2527161, 370.5406799
2: -263.1261597, 170.7345886, -265.9084778, 172.5523376, -435.6784363, 436.6430664
3: -279.9786987, 147.7608032, -282.9065552, 149.2875214, -429.2662354, 430.6673584
4: -256.7109070, 196.7213898, -259.4692688, 198.7713928, -455.4822998, 456.1906738
5: -230.1158752, 178.7985229, -232.5094452, 180.7077179, -410.8236084, 411.3079834
6: -220.1815491, 211.2866058, -222.5233765, 213.5542908, -433.7358093, 433.8099976
7: -239.5854340, 201.2774200, -242.1613312, 203.4472351, -443.0326233, 443.4386597
8: -288.6858826, 196.9381104, -291.7500916, 199.0496979, -487.7355957, 488.6882019
9: -218.0593414, 214.6311646, -220.4269104, 216.9405365, -434.9998779, 435.0580750

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2149656, upper bound: 461.2163389
time: 8.59 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2102398, upper bound: 461.2096505
time: 9.35 seconds

## BFS IS instance: IS_A2_B2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -244.0791779, 193.1385803, -242.0842743, 191.5462341, -435.6254272, 435.2228394
1: -204.0094910, 171.4488373, -202.2670288, 170.0345459, -374.0440063, 373.7158813
2: -268.1290894, 173.9264984, -265.9084778, 172.5523376, -440.6813965, 439.8349609
3: -285.3750916, 150.4561615, -282.9065552, 149.2875214, -434.6625977, 433.3627319
4: -261.6046448, 200.4210358, -259.4692688, 198.7713928, -460.3760376, 459.8903198
5: -234.5040436, 182.1492310, -232.5094452, 180.7077179, -415.2117615, 414.6586609
6: -224.3830872, 215.3204651, -222.5233765, 213.5542908, -437.9373779, 437.8438416
7: -244.1437378, 205.0820923, -242.1613312, 203.4472351, -447.5909729, 447.2433777
8: -294.1091003, 200.5506744, -291.7500916, 199.0496979, -493.1588135, 492.3007812
9: -222.2023163, 218.6915131, -220.4269104, 216.9405365, -439.1428528, 439.1183777

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B2_A2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2133710, upper bound: 461.2153275
time: 10.39 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2086098, upper bound: 461.2086098
time: 8.95 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 20.63 seconds
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2098433, upper bound: 461.2119419
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.1944120, upper bound: 461.1998551
IS_A2_B1_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2057504, upper bound: 461.2098997
IS_A2_B1_A1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2020776, upper bound: 461.2049011
IS_A2_B1_A2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2177206, upper bound: 461.2180830
IS_A2_B1_A2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2161788, upper bound: 461.2158195
IS_A2_B1_A2_A2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2183982, upper bound: 461.2167593
IS_A2_B1_A2_A2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2171483, upper bound: 461.2160555
IS_A2_B2_A2_A2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2156068, upper bound: 461.2139165
IS_A2_B2_A2_A2_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2077265, upper bound: 461.2086362
IS_A2_B2_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2106504, upper bound: 461.2140837
IS_A2_B2_A2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2060018, upper bound: 461.2075165
IS_A2_B2_A2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2149656, upper bound: 461.2163389
IS_A2_B2_A2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2102398, upper bound: 461.2096505
IS_A2_B2_A2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2133710, upper bound: 461.2153275
IS_A2_B2_A2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 20.63
Output dim: 7, lower bound: -461.2086098, upper bound: 461.2086098
Binary search (step 3): status=Status.VERIFIED, k_low=10, k_high=10, k_mid=10, eps_mid=0.0390625, abs_max=464.3514404296875
rel_dist={7: [-461.23136552831363, 461.23136552831363]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0390625
execution time: 2067.48 seconds
