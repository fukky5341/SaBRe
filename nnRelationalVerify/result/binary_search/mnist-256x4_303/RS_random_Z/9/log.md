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
execution time: IAR + LP analysis = 1.14 + 11.62 = 12.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -461.2313825, upper bound: 461.2313824


# Binary Search by BASE starts (time budget: 2687.24 seconds, max iter: 100)

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
Binary search time: 50.24 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2637.01 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2308977, upper bound: 461.2308977
time: 10.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2308977, upper bound: 461.2308977
time: 9.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.27
Output dim: 7, lower bound: -461.2308977, upper bound: 461.2308977
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.27
Output dim: 7, lower bound: -461.2308977, upper bound: 461.2308977

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1885060, upper bound: 461.1885048
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1885060, upper bound: 461.1885048
time: 7.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2279689, upper bound: 461.2279689
time: 11.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2279689, upper bound: 461.2279689
time: 9.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.63 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 21.63
Output dim: 7, lower bound: -461.1885060, upper bound: 461.1885048
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 21.63
Output dim: 7, lower bound: -461.1885060, upper bound: 461.1885048
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.63
Output dim: 7, lower bound: -461.2279689, upper bound: 461.2279689
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.63
Output dim: 7, lower bound: -461.2279689, upper bound: 461.2279689

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2279361, upper bound: 461.2279308
time: 10.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2279308, upper bound: 461.2279361
time: 8.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1985560, upper bound: 461.1985560
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1985560, upper bound: 461.1985560
time: 7.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.05 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 7, lower bound: -461.2279361, upper bound: 461.2279308
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.05
Output dim: 7, lower bound: -461.2279308, upper bound: 461.2279361
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.05
Output dim: 7, lower bound: -461.1985560, upper bound: 461.1985560
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.05
Output dim: 7, lower bound: -461.1985560, upper bound: 461.1985560

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2279308, upper bound: 461.2279308
time: 9.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2279360, upper bound: 461.2279264
time: 11.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271378, upper bound: 461.2271390
time: 9.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271378, upper bound: 461.2271390
time: 10.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.48 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 7, lower bound: -461.2279308, upper bound: 461.2279308
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 7, lower bound: -461.2279360, upper bound: 461.2279264
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 7, lower bound: -461.2271378, upper bound: 461.2271390
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 7, lower bound: -461.2271378, upper bound: 461.2271390

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1869920, upper bound: 461.1869890
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1869920, upper bound: 461.1869890
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2272688, upper bound: 461.2272537
time: 9.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2272621, upper bound: 461.2272603
time: 8.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242393
time: 9.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242393
time: 9.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251714
time: 10.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251714
time: 12.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.47 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.1869920, upper bound: 461.1869890
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.1869920, upper bound: 461.1869890
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.2272688, upper bound: 461.2272537
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.2272621, upper bound: 461.2272603
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242393
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242393
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251714
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.47
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251714

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2264598, upper bound: 461.2264528
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2264598, upper bound: 461.2264528
time: 9.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2263836, upper bound: 461.2263856
time: 9.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2263849, upper bound: 461.2263819
time: 10.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2242315, upper bound: 461.2242393
time: 10.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242369
time: 10.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2242336, upper bound: 461.2242393
time: 10.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242393
time: 9.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251710
time: 10.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251636, upper bound: 461.2251714
time: 9.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251639, upper bound: 461.2251714
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251700
time: 10.04 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 20.39 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2264598, upper bound: 461.2264528
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2264598, upper bound: 461.2264528
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2263836, upper bound: 461.2263856
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2263849, upper bound: 461.2263819
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2242315, upper bound: 461.2242393
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242369
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2242336, upper bound: 461.2242393
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2242344, upper bound: 461.2242393
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251710
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2251636, upper bound: 461.2251714
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2251639, upper bound: 461.2251714
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.39
Output dim: 7, lower bound: -461.2251646, upper bound: 461.2251700

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1619541, upper bound: 461.1618930
time: 9.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1619541, upper bound: 461.1618930
time: 9.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2229452, upper bound: 461.2229237
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2229452, upper bound: 461.2229237
time: 9.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2241810, upper bound: 461.2241857
time: 10.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2241810, upper bound: 461.2241857
time: 9.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2228887, upper bound: 461.2228956
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2228887, upper bound: 461.2228956
time: 10.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2210533, upper bound: 461.2210484
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2210533, upper bound: 461.2210484
time: 10.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1877753, upper bound: 461.1878129
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1877753, upper bound: 461.1878129
time: 8.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2107527, upper bound: 461.2108047
time: 8.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2107527, upper bound: 461.2108047
time: 9.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2164895, upper bound: 461.2164948
time: 9.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2164895, upper bound: 461.2164948
time: 10.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2238010, upper bound: 461.2238001
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2238010, upper bound: 461.2238001
time: 9.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2229155, upper bound: 461.2229250
time: 10.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2229155, upper bound: 461.2229250
time: 10.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2185902, upper bound: 461.2185946
time: 8.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2185902, upper bound: 461.2185946
time: 9.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251471, upper bound: 461.2251556
time: 9.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2251501, upper bound: 461.2251492
time: 10.50 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 23.48 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.1619541, upper bound: 461.1618930
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.1619541, upper bound: 461.1618930
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2229452, upper bound: 461.2229237
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2229452, upper bound: 461.2229237
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2241810, upper bound: 461.2241857
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2241810, upper bound: 461.2241857
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2228887, upper bound: 461.2228956
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2228887, upper bound: 461.2228956
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2210533, upper bound: 461.2210484
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2210533, upper bound: 461.2210484
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.1877753, upper bound: 461.1878129
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.1877753, upper bound: 461.1878129
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2107527, upper bound: 461.2108047
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2107527, upper bound: 461.2108047
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2164895, upper bound: 461.2164948
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2164895, upper bound: 461.2164948
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2238010, upper bound: 461.2238001
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2238010, upper bound: 461.2238001
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2229155, upper bound: 461.2229250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2229155, upper bound: 461.2229250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2185902, upper bound: 461.2185946
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2185902, upper bound: 461.2185946
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2251471, upper bound: 461.2251556
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.48
Output dim: 7, lower bound: -461.2251501, upper bound: 461.2251492

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2141890, upper bound: 461.2141542
time: 9.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2141890, upper bound: 461.2141542
time: 9.06 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 19.18 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 19.18
Output dim: 7, lower bound: -461.2141890, upper bound: 461.2141542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 19.18
Output dim: 7, lower bound: -461.2141890, upper bound: 461.2141542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2229452, upper bound: 461.2229237
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2241810, upper bound: 461.2241857
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2241810, upper bound: 461.2241857
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2228887, upper bound: 461.2228956
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2228887, upper bound: 461.2228956
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2210533, upper bound: 461.2210484
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2210533, upper bound: 461.2210484
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2238010, upper bound: 461.2238001
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2238010, upper bound: 461.2238001
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2229155, upper bound: 461.2229250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2229155, upper bound: 461.2229250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2251471, upper bound: 461.2251556
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 19.18
Output dim: 7, lower bound: -461.2251501, upper bound: 461.2251492
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=464.3514404296875
rel_dist={7: [-461.2313236619183, 461.23132366191817]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2209641, upper bound: 461.2209641
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2209641, upper bound: 461.2209641
time: 9.20 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.76
Output dim: 7, lower bound: -461.2209641, upper bound: 461.2209641
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.76
Output dim: 7, lower bound: -461.2209641, upper bound: 461.2209641

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1641821, upper bound: 461.1641821
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1641821, upper bound: 461.1641821
time: 8.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2062554, upper bound: 461.2062554
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2062554, upper bound: 461.2062554
time: 8.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.03 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 18.03
Output dim: 7, lower bound: -461.1641821, upper bound: 461.1641821
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 18.03
Output dim: 7, lower bound: -461.1641821, upper bound: 461.1641821
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 18.03
Output dim: 7, lower bound: -461.2062554, upper bound: 461.2062554
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 18.03
Output dim: 7, lower bound: -461.2062554, upper bound: 461.2062554
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=464.3514404296875
rel_dist={7: [-461.23120259765585, 461.23120260223186]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1999927, upper bound: 461.1999927
time: 9.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1999927, upper bound: 461.1999927
time: 9.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.16 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 19.16
Output dim: 7, lower bound: -461.1999927, upper bound: 461.1999927
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 19.16
Output dim: 7, lower bound: -461.1999927, upper bound: 461.1999927
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=464.3514404296875
rel_dist={7: [-461.2312546934054, 461.23125468865385]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2290429, upper bound: 461.2290429
time: 9.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2290429, upper bound: 461.2290429
time: 9.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.93 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.93
Output dim: 7, lower bound: -461.2290429, upper bound: 461.2290429
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.93
Output dim: 7, lower bound: -461.2290429, upper bound: 461.2290429

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2280665, upper bound: 461.2280665
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2280665, upper bound: 461.2280665
time: 10.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2022713, upper bound: 461.2022713
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2022713, upper bound: 461.2022713
time: 9.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.77
Output dim: 7, lower bound: -461.2280665, upper bound: 461.2280665
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.77
Output dim: 7, lower bound: -461.2280665, upper bound: 461.2280665
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 19.77
Output dim: 7, lower bound: -461.2022713, upper bound: 461.2022713
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 19.77
Output dim: 7, lower bound: -461.2022713, upper bound: 461.2022713

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2280662, upper bound: 461.2280662
time: 8.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2280662, upper bound: 461.2280662
time: 7.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2223703, upper bound: 461.2223703
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2223703, upper bound: 461.2223703
time: 9.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -461.2280662, upper bound: 461.2280662
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -461.2280662, upper bound: 461.2280662
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -461.2223703, upper bound: 461.2223703
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.85
Output dim: 7, lower bound: -461.2223703, upper bound: 461.2223703

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127228, upper bound: 461.2127228
time: 10.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127228, upper bound: 461.2127228
time: 10.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2262136, upper bound: 461.2262136
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2262136, upper bound: 461.2262136
time: 9.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2223647, upper bound: 461.2223703
time: 10.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2223703, upper bound: 461.2223647
time: 8.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1998922, upper bound: 461.1998922
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1998922, upper bound: 461.1998922
time: 7.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.2127228, upper bound: 461.2127228
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.2127228, upper bound: 461.2127228
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.2262136, upper bound: 461.2262136
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.2262136, upper bound: 461.2262136
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.2223647, upper bound: 461.2223703
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.2223703, upper bound: 461.2223647
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.1998922, upper bound: 461.1998922
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 16.23
Output dim: 7, lower bound: -461.1998922, upper bound: 461.1998922

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2172463, upper bound: 461.2172462
time: 10.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2172463, upper bound: 461.2172462
time: 9.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2217227, upper bound: 461.2217227
time: 10.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2217227, upper bound: 461.2217227
time: 10.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128017, upper bound: 461.2128009
time: 10.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128017, upper bound: 461.2128009
time: 9.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1862281, upper bound: 461.1862575
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.1862281, upper bound: 461.1862575
time: 9.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.94 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.2172463, upper bound: 461.2172462
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.2172463, upper bound: 461.2172462
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.2217227, upper bound: 461.2217227
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.2217227, upper bound: 461.2217227
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.2128017, upper bound: 461.2128009
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.2128017, upper bound: 461.2128009
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.1862281, upper bound: 461.1862575
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.94
Output dim: 7, lower bound: -461.1862281, upper bound: 461.1862575

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2118461, upper bound: 461.2118461
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2118461, upper bound: 461.2118461
time: 9.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -252.2428284, 199.5615692, -252.2428284, 199.5615692, -451.8043213, 451.8043213
1: -210.8920135, 177.2409058, -210.8920135, 177.2409058, -388.1328735, 388.1328735
2: -277.1440125, 179.7689209, -277.1440125, 179.7689209, -456.9128723, 456.9128723
3: -294.8527222, 155.5875702, -294.8527222, 155.5875702, -450.4403076, 450.4403076
4: -270.4500427, 207.2221527, -270.4500427, 207.2221527, -477.6721802, 477.6721802
5: -242.3452606, 188.3783875, -242.3452606, 188.3783875, -430.7236328, 430.7236328
6: -231.8721466, 222.5393829, -231.8721466, 222.5393829, -454.4115295, 454.4115295
7: -252.3833771, 211.9680939, -252.3833771, 211.9680939, -464.3514404, 464.3514404
8: -304.0357361, 207.4447021, -304.0357361, 207.4447021, -511.4804382, 511.4803772
9: -229.7125092, 226.0945892, -229.7125092, 226.0945892, -455.8070984, 455.8070984

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2025241, upper bound: 461.2025218
time: 8.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2025241, upper bound: 461.2025218
time: 8.98 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 18.98 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 18.98
Output dim: 7, lower bound: -461.2118461, upper bound: 461.2118461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 18.98
Output dim: 7, lower bound: -461.2118461, upper bound: 461.2118461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 18.98
Output dim: 7, lower bound: -461.2025241, upper bound: 461.2025218
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 18.98
Output dim: 7, lower bound: -461.2025241, upper bound: 461.2025218
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=464.3514404296875
rel_dist={7: [-461.23129454308656, 461.23129452244643]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 1036.64 seconds
