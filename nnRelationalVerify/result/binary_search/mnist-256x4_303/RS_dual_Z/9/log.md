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
execution time: IAR + LP analysis = 1.18 + 11.64 = 12.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -461.2313825, upper bound: 461.2313824


# Binary Search by BASE starts (time budget: 2687.18 seconds, max iter: 100)

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
Binary search time: 49.68 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2637.49 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
time: 8.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.33
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.33
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270913

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270876, upper bound: 461.2270913
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270877
time: 8.70 seconds

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270877, upper bound: 461.2270913
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270877
time: 8.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 7, lower bound: -461.2270876, upper bound: 461.2270913
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270877
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 7, lower bound: -461.2270877, upper bound: 461.2270913
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 7, lower bound: -461.2270913, upper bound: 461.2270877

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270544, upper bound: 461.2270666
time: 10.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270634, upper bound: 461.2270571
time: 9.00 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270571, upper bound: 461.2270634
time: 9.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270666, upper bound: 461.2270544
time: 8.83 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270544, upper bound: 461.2270666
time: 10.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270634, upper bound: 461.2270571
time: 8.76 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270571, upper bound: 461.2270634
time: 9.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2270666, upper bound: 461.2270544
time: 8.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270544, upper bound: 461.2270666
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270634, upper bound: 461.2270571
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270571, upper bound: 461.2270634
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270666, upper bound: 461.2270544
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270544, upper bound: 461.2270666
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270634, upper bound: 461.2270571
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270571, upper bound: 461.2270634
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.86
Output dim: 7, lower bound: -461.2270666, upper bound: 461.2270544

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258371, upper bound: 461.2258891
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258732, upper bound: 461.2258555
time: 8.78 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258497, upper bound: 461.2258793
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258811, upper bound: 461.2258390
time: 9.86 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258390, upper bound: 461.2258811
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258793, upper bound: 461.2258497
time: 8.49 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258555, upper bound: 461.2258732
time: 10.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258891, upper bound: 461.2258371
time: 9.79 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258371, upper bound: 461.2258891
time: 8.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258732, upper bound: 461.2258555
time: 10.06 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258497, upper bound: 461.2258793
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258811, upper bound: 461.2258390
time: 8.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258390, upper bound: 461.2258811
time: 10.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258793, upper bound: 461.2258497
time: 9.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258555, upper bound: 461.2258732
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258891, upper bound: 461.2258371
time: 9.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258371, upper bound: 461.2258891
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258732, upper bound: 461.2258555
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258497, upper bound: 461.2258793
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258811, upper bound: 461.2258390
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258390, upper bound: 461.2258811
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258793, upper bound: 461.2258497
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258555, upper bound: 461.2258732
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258891, upper bound: 461.2258371
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258371, upper bound: 461.2258891
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258732, upper bound: 461.2258555
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258497, upper bound: 461.2258793
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258811, upper bound: 461.2258390
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258390, upper bound: 461.2258811
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258793, upper bound: 461.2258497
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258555, upper bound: 461.2258732
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.81
Output dim: 7, lower bound: -461.2258891, upper bound: 461.2258371

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
time: 8.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
time: 8.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
time: 8.33 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
time: 8.69 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
time: 9.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
time: 9.37 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
time: 9.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
time: 8.68 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
time: 8.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
time: 9.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
time: 9.33 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
time: 8.66 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
time: 10.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
time: 10.01 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
time: 8.94 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
time: 8.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
time: 8.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
time: 9.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
time: 8.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
time: 10.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
time: 8.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
time: 9.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
time: 9.43 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126769, upper bound: 461.2127293
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127095, upper bound: 461.2126997
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126975, upper bound: 461.2127145
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127224, upper bound: 461.2126805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126805, upper bound: 461.2127224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127145, upper bound: 461.2126975
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2126997, upper bound: 461.2127095
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.70
Output dim: 7, lower bound: -461.2127293, upper bound: 461.2126769
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=464.3514404296875
rel_dist={7: [-461.2313236619183, 461.23132366191817]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271332
time: 8.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271332
time: 8.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.42
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271332
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.42
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271332

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271287, upper bound: 461.2271332
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271287
time: 9.13 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271287, upper bound: 461.2271332
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271287
time: 10.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.97 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -461.2271287, upper bound: 461.2271332
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271287
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -461.2271287, upper bound: 461.2271332
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.97
Output dim: 7, lower bound: -461.2271332, upper bound: 461.2271287

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271051, upper bound: 461.2271223
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271157, upper bound: 461.2271095
time: 8.40 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271095, upper bound: 461.2271157
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271223, upper bound: 461.2271051
time: 8.57 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271051, upper bound: 461.2271223
time: 9.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271157, upper bound: 461.2271095
time: 8.43 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271095, upper bound: 461.2271157
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271223, upper bound: 461.2271051
time: 9.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271051, upper bound: 461.2271223
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271157, upper bound: 461.2271095
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271095, upper bound: 461.2271157
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271223, upper bound: 461.2271051
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271051, upper bound: 461.2271223
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271157, upper bound: 461.2271095
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271095, upper bound: 461.2271157
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 7, lower bound: -461.2271223, upper bound: 461.2271051

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258799, upper bound: 461.2259528
time: 8.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259351, upper bound: 461.2259064
time: 9.71 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259008, upper bound: 461.2259390
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259463, upper bound: 461.2258817
time: 9.77 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258817, upper bound: 461.2259463
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259390, upper bound: 461.2259008
time: 8.96 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259064, upper bound: 461.2259351
time: 8.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259528, upper bound: 461.2258799
time: 9.32 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258799, upper bound: 461.2259528
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259351, upper bound: 461.2259064
time: 9.30 seconds

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258799, upper bound: 461.2259390
time: 9.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259463, upper bound: 461.2258817
time: 8.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2258817, upper bound: 461.2259463
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259390, upper bound: 461.2259008
time: 7.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259064, upper bound: 461.2259351
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259528, upper bound: 461.2258799
time: 9.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2258799, upper bound: 461.2259528
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259351, upper bound: 461.2259064
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259008, upper bound: 461.2259390
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259463, upper bound: 461.2258817
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2258817, upper bound: 461.2259463
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259390, upper bound: 461.2259008
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259064, upper bound: 461.2259351
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259528, upper bound: 461.2258799
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2258799, upper bound: 461.2259528
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259351, upper bound: 461.2259064
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2258799, upper bound: 461.2259390
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259463, upper bound: 461.2258817
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2258817, upper bound: 461.2259463
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259390, upper bound: 461.2259008
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259064, upper bound: 461.2259351
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.35
Output dim: 7, lower bound: -461.2259528, upper bound: 461.2258799

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
time: 8.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
time: 8.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
time: 8.75 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
time: 8.60 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
time: 9.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
time: 9.07 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
time: 8.43 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127809, upper bound: 461.2127591
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127591
time: 9.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
time: 8.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
time: 9.06 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
time: 8.11 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
time: 9.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
time: 8.89 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
time: 10.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
time: 10.83 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
time: 8.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
time: 7.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127810, upper bound: 461.2127591
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127810, upper bound: 461.2127591
time: 8.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
time: 8.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
time: 9.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127809, upper bound: 461.2127591
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127591
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127285, upper bound: 461.2128033
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127742, upper bound: 461.2127612
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127591, upper bound: 461.2127810
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127924, upper bound: 461.2127334
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127334, upper bound: 461.2127924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127810, upper bound: 461.2127591
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127810, upper bound: 461.2127591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2127612, upper bound: 461.2127742
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.55
Output dim: 7, lower bound: -461.2128033, upper bound: 461.2127285
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=464.3514404296875
rel_dist={7: [-461.2313563734555, 461.2313561033127]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271599
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271599
time: 7.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.40
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271599
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.40
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271599

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271547, upper bound: 461.2271599
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271547
time: 8.05 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271547, upper bound: 461.2271599
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271547
time: 8.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2271547, upper bound: 461.2271599
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271547
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2271547, upper bound: 461.2271599
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 7, lower bound: -461.2271599, upper bound: 461.2271547

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271346, upper bound: 461.2271545
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271461, upper bound: 461.2271393
time: 9.51 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271393, upper bound: 461.2271461
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271545, upper bound: 461.2271345
time: 7.95 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271346, upper bound: 461.2271545
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271461, upper bound: 461.2271393
time: 8.02 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271393, upper bound: 461.2271461
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271545, upper bound: 461.2271346
time: 7.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271346, upper bound: 461.2271545
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271461, upper bound: 461.2271393
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271393, upper bound: 461.2271461
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271545, upper bound: 461.2271345
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271346, upper bound: 461.2271545
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271461, upper bound: 461.2271393
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271393, upper bound: 461.2271461
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.49
Output dim: 7, lower bound: -461.2271545, upper bound: 461.2271346

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259060, upper bound: 461.2259905
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259721, upper bound: 461.2259354
time: 9.77 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259310, upper bound: 461.2259731
time: 9.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259858, upper bound: 461.2259077
time: 8.63 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259077, upper bound: 461.2259858
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259731, upper bound: 461.2259310
time: 9.15 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259354, upper bound: 461.2259721
time: 9.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259060
time: 8.88 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259060, upper bound: 461.2259905
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259721, upper bound: 461.2259354
time: 10.60 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259310, upper bound: 461.2259731
time: 9.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259858, upper bound: 461.2259077
time: 9.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259077, upper bound: 461.2259858
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259731, upper bound: 461.2259310
time: 9.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259354, upper bound: 461.2259721
time: 9.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259060
time: 8.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259060, upper bound: 461.2259905
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259721, upper bound: 461.2259354
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259310, upper bound: 461.2259731
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259858, upper bound: 461.2259077
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259077, upper bound: 461.2259858
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259731, upper bound: 461.2259310
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259354, upper bound: 461.2259721
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259060
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259060, upper bound: 461.2259905
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259721, upper bound: 461.2259354
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259310, upper bound: 461.2259731
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259858, upper bound: 461.2259077
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259077, upper bound: 461.2259858
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259731, upper bound: 461.2259310
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259354, upper bound: 461.2259721
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.14
Output dim: 7, lower bound: -461.2259905, upper bound: 461.2259060

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
time: 8.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
time: 11.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
time: 8.97 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
time: 7.61 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
time: 8.69 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
time: 9.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
time: 9.36 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
time: 9.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
time: 8.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
time: 9.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
time: 8.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2127610
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128490, upper bound: 461.2127610
time: 8.99 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
time: 7.80 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
time: 9.85 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
time: 8.92 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
time: 10.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
time: 8.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
time: 7.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
time: 8.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
time: 8.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
time: 12.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
time: 8.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128489, upper bound: 461.2127610
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128489, upper bound: 461.2127610
time: 8.70 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2127610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128490, upper bound: 461.2127610
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127610, upper bound: 461.2128490
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128154, upper bound: 461.2128001
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127975, upper bound: 461.2128224
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128348, upper bound: 461.2127669
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2127669, upper bound: 461.2128348
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128224, upper bound: 461.2127975
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128001, upper bound: 461.2128154
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128489, upper bound: 461.2127610
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.49
Output dim: 7, lower bound: -461.2128489, upper bound: 461.2127610
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=464.3514404296875
rel_dist={7: [-461.2313740276876, 461.23137391660043]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271716
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271715
time: 7.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.36 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.36
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271716
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.36
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271715

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271672, upper bound: 461.2271716
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271671
time: 8.48 seconds

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271672, upper bound: 461.2271716
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271672
time: 8.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.45
Output dim: 7, lower bound: -461.2271672, upper bound: 461.2271716
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.45
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271671
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.45
Output dim: 7, lower bound: -461.2271672, upper bound: 461.2271716
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.45
Output dim: 7, lower bound: -461.2271716, upper bound: 461.2271672

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

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271487, upper bound: 461.2271675
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271598, upper bound: 461.2271530
time: 8.13 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271530, upper bound: 461.2271598
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271675, upper bound: 461.2271487
time: 8.68 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271487, upper bound: 461.2271675
time: 7.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271598, upper bound: 461.2271530
time: 8.04 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271530, upper bound: 461.2271598
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2271675, upper bound: 461.2271487
time: 8.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271487, upper bound: 461.2271675
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271598, upper bound: 461.2271530
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271530, upper bound: 461.2271598
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271675, upper bound: 461.2271487
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271487, upper bound: 461.2271675
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271598, upper bound: 461.2271530
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271530, upper bound: 461.2271598
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.59
Output dim: 7, lower bound: -461.2271675, upper bound: 461.2271487

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259188, upper bound: 461.2260057
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259874, upper bound: 461.2259489
time: 9.40 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259450, upper bound: 461.2259886
time: 10.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2260019, upper bound: 461.2259198
time: 8.05 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259198, upper bound: 461.2260019
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259886, upper bound: 461.2259450
time: 7.81 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259490, upper bound: 461.2259874
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2260055, upper bound: 461.2259188
time: 7.58 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259188, upper bound: 461.2260055
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259188, upper bound: 461.2259489
time: 9.14 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259450, upper bound: 461.2259886
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2260019, upper bound: 461.2259198
time: 7.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259198, upper bound: 461.2260019
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259886, upper bound: 461.2259450
time: 8.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2259490, upper bound: 461.2259874
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -461.2260055, upper bound: 461.2259188
time: 7.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259188, upper bound: 461.2260057
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259874, upper bound: 461.2259489
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259450, upper bound: 461.2259886
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2260019, upper bound: 461.2259198
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259198, upper bound: 461.2260019
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259886, upper bound: 461.2259450
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259490, upper bound: 461.2259874
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2260055, upper bound: 461.2259188
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259188, upper bound: 461.2260055
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259188, upper bound: 461.2259489
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259450, upper bound: 461.2259886
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2260019, upper bound: 461.2259198
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259198, upper bound: 461.2260019
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259886, upper bound: 461.2259450
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2259490, upper bound: 461.2259874
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 7, lower bound: -461.2260055, upper bound: 461.2259188

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
time: 7.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
time: 8.18 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
time: 6.92 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
time: 7.42 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
time: 8.28 seconds

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128342
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128342
time: 6.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
time: 7.54 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
time: 7.57 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
time: 10.56 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
time: 7.40 seconds

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
time: 7.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
time: 7.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
time: 9.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
time: 8.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128342
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128341
time: 7.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
time: 7.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
time: 7.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128342
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127769, upper bound: 461.2128717
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128342, upper bound: 461.2128186
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128158, upper bound: 461.2128426
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128554, upper bound: 461.2127829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2127829, upper bound: 461.2128554
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128426, upper bound: 461.2128158
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128342
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128186, upper bound: 461.2128341
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.26
Output dim: 7, lower bound: -461.2128717, upper bound: 461.2127769
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=464.3514404296875
rel_dist={7: [-461.23138251780557, 461.2313824043131]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 2349.78 seconds
