## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 27.1733048946
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570)
1: (-16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204)
2: (-27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886)
3: (-24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909)
4: (-24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434)
5: (-18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721)
6: (-19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750)
7: (-22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555)
8: (-25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507)
9: (-17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 16.35 = 17.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2006483, upper bound: 27.2006483


# Binary Search by BASE starts (time budget: 1982.41 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=36.862388610839844
rel_dist={2: [-27.200554238179546, 27.20055423068736]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=36.862388610839844
rel_dist={2: [-27.20050538908776, 27.200505386462353]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=36.862388610839844
rel_dist={2: [-27.200440708568877, 27.200440707169562]}

## Binary Search Result
Binary search time: 28.55 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1953.86 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976940, upper bound: 27.1971771
time: 6.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1988318, upper bound: 27.1988318
time: 5.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.05
Output dim: 2, lower bound: -27.1976940, upper bound: 27.1971771
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.05
Output dim: 2, lower bound: -27.1988318, upper bound: 27.1988318

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -20.5011635, 15.6042356, -33.3413200, 33.9394302
1: -14.1783390, 11.9624004, -16.4622440, 13.9153538, -28.0936890, 28.4246445
2: -23.5818024, 7.6063967, -26.9467468, 9.4406891, -33.0224915, 34.5531387
3: -21.0418549, 9.4901848, -24.3798428, 11.1349821, -32.1768379, 33.8700256
4: -21.2769165, 12.8763294, -24.4749527, 14.9834261, -36.2603416, 37.3512802
5: -15.8263702, 13.6713247, -18.4286251, 15.7944098, -31.6207809, 32.0999489
6: -16.8846321, 14.4183712, -19.6420097, 16.7497520, -33.6343803, 34.0603790
7: -19.0637054, 13.8504381, -22.0094910, 16.2135944, -35.2772980, 35.8599243
8: -21.9471779, 12.5008354, -25.4139977, 14.5830660, -36.5302391, 37.9148254
9: -15.2502155, 17.9744301, -17.7359314, 20.6399536, -35.8901672, 35.7103577

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966865, upper bound: 27.1966865
time: 3.16 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966865, upper bound: 27.1971771
time: 8.84 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -20.6256313, 15.7058964, -34.9872665, 35.2584877
1: -15.4504633, 13.0406036, -16.5659409, 14.0063839, -29.4568481, 29.6065445
2: -25.4806252, 8.5678139, -27.0929527, 9.5397377, -35.0203552, 35.6607628
3: -22.9028149, 10.3804493, -24.5305328, 11.2151241, -34.1179390, 34.9109802
4: -23.0672989, 14.0238628, -24.6155357, 15.0844221, -38.1517220, 38.6393967
5: -17.2664795, 14.8502493, -18.5497799, 15.8910637, -33.1575432, 33.4000206
6: -18.4147415, 15.7105303, -19.7680035, 16.8584766, -35.2732162, 35.4785347
7: -20.7114697, 15.1501713, -22.1425896, 16.3263206, -37.0377884, 37.2927628
8: -23.8846569, 13.6342249, -25.5701237, 14.6839333, -38.5685844, 39.2043495
9: -16.6258526, 19.4761505, -17.8513336, 20.7563324, -37.3821869, 37.3274841

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971771, upper bound: 27.1976940
time: 4.26 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971771, upper bound: 27.1988318
time: 6.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 12.30 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 12.30
Output dim: 2, lower bound: -27.1966865, upper bound: 27.1966865
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 12.30
Output dim: 2, lower bound: -27.1966865, upper bound: 27.1971771
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 12.30
Output dim: 2, lower bound: -27.1971771, upper bound: 27.1976940
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 12.30
Output dim: 2, lower bound: -27.1971771, upper bound: 27.1988318

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -17.7370834, 13.4382677, -31.1753502, 31.1753502
1: -14.1783390, 11.9624004, -14.1783390, 11.9624004, -26.1407394, 26.1407394
2: -23.5818024, 7.6063967, -23.5818024, 7.6063967, -31.1881962, 31.1881981
3: -21.0418549, 9.4901848, -21.0418549, 9.4901848, -30.5320396, 30.5320396
4: -21.2769165, 12.8763294, -21.2769165, 12.8763294, -34.1532440, 34.1532440
5: -15.8263702, 13.6713247, -15.8263702, 13.6713247, -29.4976959, 29.4976959
6: -16.8846321, 14.4183712, -16.8846321, 14.4183712, -31.3030014, 31.3030033
7: -19.0637054, 13.8504381, -19.0637054, 13.8504381, -32.9141426, 32.9141426
8: -21.9471779, 12.5008354, -21.9471779, 12.5008354, -34.4480057, 34.4480057
9: -15.2502155, 17.9744301, -15.2502155, 17.9744301, -33.2246475, 33.2246475

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966860, upper bound: 27.1966861
time: 6.23 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1966861
time: 3.83 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -19.2813759, 14.6328573, -32.3699417, 32.7196426
1: -14.1783390, 11.9624004, -15.4504633, 13.0406036, -27.2189426, 27.4128647
2: -23.5818024, 7.6063967, -25.4806252, 8.5678139, -32.1496048, 33.0870132
3: -21.0418549, 9.4901848, -22.9028149, 10.3804493, -31.4223042, 32.3929977
4: -21.2769165, 12.8763294, -23.0672989, 14.0238628, -35.3007812, 35.9436264
5: -15.8263702, 13.6713247, -17.2664795, 14.8502493, -30.6766167, 30.9378052
6: -16.8846321, 14.4183712, -18.4147415, 15.7105303, -32.5951614, 32.8331146
7: -19.0637054, 13.8504381, -20.7114697, 15.1501713, -34.2138748, 34.5619049
8: -21.9471779, 12.5008354, -23.8846569, 13.6342249, -35.5813980, 36.3854866
9: -15.2502155, 17.9744301, -16.6258526, 19.4761505, -34.7263641, 34.6002808

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966860, upper bound: 27.1971771
time: 4.93 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1971743
time: 3.15 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -17.7370834, 13.4382677, -32.7196388, 32.3699417
1: -15.4504633, 13.0406036, -14.1783390, 11.9624004, -27.4128647, 27.2189426
2: -25.4806252, 8.5678139, -23.5818024, 7.6063967, -33.0870171, 32.1496124
3: -22.9028149, 10.3804493, -21.0418549, 9.4901848, -32.3929977, 31.4223042
4: -23.0672989, 14.0238628, -21.2769165, 12.8763294, -35.9436264, 35.3007812
5: -17.2664795, 14.8502493, -15.8263702, 13.6713247, -30.9378052, 30.6766186
6: -18.4147415, 15.7105303, -16.8846321, 14.4183712, -32.8331146, 32.5951614
7: -20.7114697, 15.1501713, -19.0637054, 13.8504381, -34.5619087, 34.2138748
8: -23.8846569, 13.6342249, -21.9471779, 12.5008354, -36.3854866, 35.5813980
9: -16.6258526, 19.4761505, -15.2502155, 17.9744301, -34.6002808, 34.7263641

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971661, upper bound: 27.1976857
time: 3.76 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1976940
time: 6.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -19.2813759, 14.6328573, -33.9142342, 33.9142342
1: -15.4504633, 13.0406036, -15.4504633, 13.0406036, -28.4910660, 28.4910660
2: -25.4806252, 8.5678139, -25.4806252, 8.5678139, -34.0484314, 34.0484276
3: -22.9028149, 10.3804493, -22.9028149, 10.3804493, -33.2832565, 33.2832565
4: -23.0672989, 14.0238628, -23.0672989, 14.0238628, -37.0911636, 37.0911636
5: -17.2664795, 14.8502493, -17.2664795, 14.8502493, -32.1167297, 32.1167297
6: -18.4147415, 15.7105303, -18.4147415, 15.7105303, -34.1252670, 34.1252708
7: -20.7114697, 15.1501713, -20.7114697, 15.1501713, -35.8616409, 35.8616409
8: -23.8846569, 13.6342249, -23.8846569, 13.6342249, -37.5188828, 37.5188789
9: -16.6258526, 19.4761505, -16.6258526, 19.4761505, -36.1020050, 36.1020050

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971661, upper bound: 27.1988185
time: 5.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1988318
time: 5.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 12.80 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1966860, upper bound: 27.1966861
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1966861
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1966860, upper bound: 27.1971771
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1971743
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1971661, upper bound: 27.1976857
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1976940
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1971661, upper bound: 27.1988185
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.80
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1988318

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.9330330, 12.0862713, -17.5979614, 13.3315668, -29.2645988, 29.6842327
1: -12.7195988, 10.7698059, -14.0653410, 11.8687382, -24.5883312, 24.8351440
2: -21.2332172, 6.7949381, -23.4031868, 7.5335503, -28.7667675, 30.1981163
3: -18.8858185, 8.5411921, -20.8771400, 9.4145222, -28.3003407, 29.4183311
4: -19.1408081, 11.5842705, -21.1148453, 12.7738514, -31.9146500, 32.6991158
5: -14.2070770, 12.3037930, -15.6989260, 13.5662384, -27.7733154, 28.0027161
6: -15.1586037, 12.9699535, -16.7482452, 14.3058643, -29.4644680, 29.7181988
7: -17.1362457, 12.4460955, -18.9156685, 13.7387476, -30.8749886, 31.3617630
8: -19.7245827, 11.2484522, -21.7738228, 12.4001713, -32.1247559, 33.0222740
9: -13.7096291, 16.1677799, -15.1298828, 17.8359489, -31.5455780, 31.2976627

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1931067, upper bound: 27.1922610
time: 6.94 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908211, upper bound: 27.1908548
time: 5.09 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.3086624, 13.1110954, -17.7370834, 13.4382677, -30.7469292, 30.8481750
1: -13.8301620, 11.6761665, -14.1783390, 11.9624004, -25.7925625, 25.8545036
2: -23.0391483, 7.3856821, -23.5818024, 7.6063967, -30.6455441, 30.9674835
3: -20.5315437, 9.2545481, -21.0418549, 9.4901848, -30.0217285, 30.2964020
4: -20.7801514, 12.5645094, -21.2769165, 12.8763294, -33.6564789, 33.8414268
5: -15.4340830, 13.3476000, -15.8263702, 13.6713247, -29.1054039, 29.1739693
6: -16.4678898, 14.0741301, -16.8846321, 14.4183712, -30.8862534, 30.9587631
7: -18.6089172, 13.5067215, -19.0637054, 13.8504381, -32.4593544, 32.5704269
8: -21.4197464, 12.1960192, -21.9471779, 12.5008354, -33.9205818, 34.1431923
9: -14.8818188, 17.5486946, -15.2502155, 17.9744301, -32.8562469, 32.7989120

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1966860
time: 4.20 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1966861
time: 5.41 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15.9330330, 12.0862713, -19.1077766, 14.4968386, -30.4298706, 31.1940479
1: -12.7195988, 10.7698059, -15.3074636, 12.9193325, -25.6389313, 26.0772705
2: -21.2332172, 6.7949381, -25.2624855, 8.4638577, -29.6970749, 32.0574188
3: -18.8858185, 8.5411921, -22.6914787, 10.2798901, -29.1657066, 31.2326698
4: -19.1408081, 11.5842705, -22.8639374, 13.8939514, -33.0347481, 34.4482079
5: -14.2070770, 12.3037930, -17.1053543, 14.7163486, -28.9234200, 29.4091473
6: -15.1586037, 12.9699535, -18.2426910, 15.5668945, -30.7254887, 31.2126446
7: -17.1362457, 12.4460955, -20.5282707, 15.0053368, -32.1415825, 32.9743652
8: -19.7245827, 11.2484522, -23.6676903, 13.5038891, -33.2284698, 34.9161415
9: -13.7096291, 16.1677799, -16.4712181, 19.3050041, -33.0146332, 32.6389923

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1970994, upper bound: 27.1967596
time: 3.79 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1975091, upper bound: 27.1970317
time: 49.28 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17.3086624, 13.1110954, -19.2813759, 14.6328573, -31.9415207, 32.3924637
1: -13.8301620, 11.6761665, -15.4504633, 13.0406036, -26.8707619, 27.1266289
2: -23.0391483, 7.3856821, -25.4806252, 8.5678139, -31.6069622, 32.8662987
3: -20.5315437, 9.2545481, -22.9028149, 10.3804493, -30.9119930, 32.1573639
4: -20.7801514, 12.5645094, -23.0672989, 14.0238628, -34.8040085, 35.6318092
5: -15.4340830, 13.3476000, -17.2664795, 14.8502493, -30.2843285, 30.6140785
6: -16.4678898, 14.0741301, -18.4147415, 15.7105303, -32.1784134, 32.4888725
7: -18.6089172, 13.5067215, -20.7114697, 15.1501713, -33.7590866, 34.2181931
8: -21.4197464, 12.1960192, -23.8846569, 13.6342249, -35.0539703, 36.0806732
9: -14.8818188, 17.5486946, -16.6258526, 19.4761505, -34.3579636, 34.1745453

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976857, upper bound: 27.1971662
time: 4.54 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976857, upper bound: 27.1971743
time: 6.65 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.3664093, 13.1840649, -17.5979614, 13.3315668, -30.6979752, 30.7820263
1: -13.8961716, 11.7444916, -14.0653410, 11.8687382, -25.7649059, 25.8098297
2: -23.0205383, 7.6025352, -23.4031868, 7.5335503, -30.5540791, 31.0057182
3: -20.5968895, 9.3359022, -20.8771400, 9.4145222, -30.0114117, 30.2130375
4: -20.8112106, 12.6294336, -21.1148453, 12.7738514, -33.5850563, 33.7442780
5: -15.5342627, 13.3937263, -15.6989260, 13.5662384, -29.1004963, 29.0926514
6: -16.5527077, 14.1519051, -16.7482452, 14.3058643, -30.8585720, 30.9001503
7: -18.6795216, 13.6284981, -18.9156685, 13.7387476, -32.4182625, 32.5441666
8: -21.5059509, 12.2588511, -21.7738228, 12.4001713, -33.9061203, 34.0326729
9: -14.9652519, 17.5726089, -15.1298828, 17.8359489, -32.8011932, 32.7024918

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1935424, upper bound: 27.1922610
time: 7.96 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1913170, upper bound: 27.1918547
time: 4.33 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18.7227573, 14.2010355, -17.7370834, 13.4382677, -32.1610184, 31.9381180
1: -14.9935465, 12.6504822, -14.1783390, 11.9624004, -26.9559479, 26.8288193
2: -24.7911720, 8.2266684, -23.5818024, 7.6063967, -32.3975639, 31.8084679
3: -22.2230797, 10.0535164, -21.0418549, 9.4901848, -31.7132645, 31.0953712
4: -22.4221344, 13.6065416, -21.2769165, 12.8763294, -35.2984619, 34.8834572
5: -16.7527390, 14.4238949, -15.8263702, 13.6713247, -30.4240608, 30.2502651
6: -17.8588409, 15.2488451, -16.8846321, 14.4183712, -32.2772141, 32.1334763
7: -20.1265373, 14.6859989, -19.0637054, 13.8504381, -33.9769707, 33.7497025
8: -23.1871510, 13.2136288, -21.9471779, 12.5008354, -35.6879845, 35.1608009
9: -16.1301155, 18.9302635, -15.2502155, 17.9744301, -34.1045456, 34.1804733

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1976874
time: 4.41 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1976940
time: 5.26 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.3664093, 13.1840649, -19.1077766, 14.4968386, -31.8632469, 32.2918396
1: -13.8961716, 11.7444916, -15.3074636, 12.9193325, -26.8155041, 27.0519562
2: -23.0205383, 7.6025352, -25.2624855, 8.4638577, -31.4843864, 32.8650169
3: -20.5968895, 9.3359022, -22.6914787, 10.2798901, -30.8767776, 32.0273819
4: -20.8112106, 12.6294336, -22.8639374, 13.8939514, -34.7051544, 35.4933701
5: -15.5342627, 13.3937263, -17.1053543, 14.7163486, -30.2505989, 30.4990807
6: -16.5527077, 14.1519051, -18.2426910, 15.5668945, -32.1195984, 32.3945961
7: -18.6795216, 13.6284981, -20.5282707, 15.0053368, -33.6848602, 34.1567688
8: -21.5059509, 12.2588511, -23.6676903, 13.5038891, -35.0098419, 35.9265404
9: -14.9652519, 17.5726089, -16.4712181, 19.3050041, -34.2702560, 34.0438194

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1983224, upper bound: 27.1984445
time: 5.21 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987206, upper bound: 27.1987578
time: 4.70 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18.7227573, 14.2010355, -19.2813759, 14.6328573, -33.3556137, 33.4824104
1: -14.9935465, 12.6504822, -15.4504633, 13.0406036, -28.0341454, 28.1009445
2: -24.7911720, 8.2266684, -25.4806252, 8.5678139, -33.3589783, 33.7072830
3: -22.2230797, 10.0535164, -22.9028149, 10.3804493, -32.6035156, 32.9563293
4: -22.4221344, 13.6065416, -23.0672989, 14.0238628, -36.4459953, 36.6738396
5: -16.7527390, 14.4238949, -17.2664795, 14.8502493, -31.6029892, 31.6903744
6: -17.8588409, 15.2488451, -18.4147415, 15.7105303, -33.5693703, 33.6635857
7: -20.1265373, 14.6859989, -20.7114697, 15.1501713, -35.2767029, 35.3974686
8: -23.1871510, 13.2136288, -23.8846569, 13.6342249, -36.8213730, 37.0982819
9: -16.1301155, 18.9302635, -16.6258526, 19.4761505, -35.6062622, 35.5561142

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1988059, upper bound: 27.1988216
time: 5.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1988059, upper bound: 27.1988318
time: 4.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 11.72 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1931067, upper bound: 27.1922610
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1908211, upper bound: 27.1908548
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1966860
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1966861, upper bound: 27.1966861
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1970994, upper bound: 27.1967596
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1975091, upper bound: 27.1970317
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1976857, upper bound: 27.1971662
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1976857, upper bound: 27.1971743
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1935424, upper bound: 27.1922610
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1913170, upper bound: 27.1918547
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1976874
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1971743, upper bound: 27.1976940
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1983224, upper bound: 27.1984445
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1987206, upper bound: 27.1987578
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1988059, upper bound: 27.1988216
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 11.72
Output dim: 2, lower bound: -27.1988059, upper bound: 27.1988318

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.9330330, 12.0862713, -17.0256939, 12.9024696, -28.8355026, 29.1119652
1: -12.7195988, 10.7698059, -13.6007490, 11.4886112, -24.2082100, 24.3705559
2: -21.2332172, 6.7949381, -22.6536369, 7.2771072, -28.5103245, 29.4485703
3: -18.8858185, 8.5411921, -20.1921387, 9.1147385, -28.0005569, 28.7333298
4: -19.1408081, 11.5842705, -20.4353695, 12.3647251, -31.5055275, 32.0196381
5: -14.2070770, 12.3037930, -15.1843166, 13.1317902, -27.3388672, 27.4881096
6: -15.1586037, 12.9699535, -16.2016716, 13.8452625, -29.0038662, 29.1716232
7: -17.1362457, 12.4460955, -18.3026276, 13.2946835, -30.4309292, 30.7487221
8: -19.7245827, 11.2484522, -21.0666389, 12.0034809, -31.7280636, 32.3150864
9: -13.7096291, 16.1677799, -14.6406431, 17.2622051, -30.9718342, 30.8084221

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908211, upper bound: 27.1908548
time: 3.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908211, upper bound: 27.1908548
time: 3.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.7699070, 11.9646492, -20.0851307, 15.2364426, -31.0063496, 32.0497818
1: -12.5871677, 10.6615982, -16.1005745, 13.5622196, -26.1493874, 26.7621727
2: -21.0193863, 6.7231970, -26.5709457, 8.8136921, -29.8330765, 33.2941399
3: -18.6903877, 8.4563627, -23.8911934, 10.7784395, -29.4688263, 32.3475571
4: -18.9471169, 11.4680748, -24.0719776, 14.5925684, -33.5396843, 35.5400543
5: -14.0615606, 12.1798630, -18.0007496, 15.4684067, -29.5299683, 30.1806126
6: -15.0033865, 12.8387241, -19.1505184, 16.3435268, -31.3469048, 31.9892406
7: -16.9613152, 12.3198795, -21.5906277, 15.7344074, -32.6957207, 33.9105072
8: -19.5232105, 11.1358242, -24.8927250, 14.1446533, -33.6678619, 36.0285454
9: -13.5702267, 16.0045109, -17.3022385, 20.3124542, -33.8826752, 33.3067474

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907258, upper bound: 27.1907534
time: 3.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907309, upper bound: 27.1907566
time: 4.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.3086624, 13.1110954, -15.9330330, 12.0862713, -29.3949337, 29.0441246
1: -13.8301620, 11.6761665, -12.7195988, 10.7698059, -24.5999680, 24.3957596
2: -23.0391483, 7.3856821, -21.2332172, 6.7949381, -29.8340874, 28.6188984
3: -20.5315437, 9.2545481, -18.8858185, 8.5411921, -29.0727348, 28.1403656
4: -20.7801514, 12.5645094, -19.1408081, 11.5842705, -32.3644218, 31.7053108
5: -15.4340830, 13.3476000, -14.2070770, 12.3037930, -27.7378731, 27.5546761
6: -16.4678898, 14.0741301, -15.1586037, 12.9699535, -29.4378357, 29.2327347
7: -18.6089172, 13.5067215, -17.1362457, 12.4460955, -31.0550117, 30.6429653
8: -21.4197464, 12.1960192, -19.7245827, 11.2484522, -32.6681976, 31.9206009
9: -14.8818188, 17.5486946, -13.7096291, 16.1677799, -31.0495949, 31.2583237

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1922204, upper bound: 27.1930815
time: 6.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
time: 5.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.3086624, 13.1110954, -17.3086624, 13.1110954, -30.4197540, 30.4197502
1: -13.8301620, 11.6761665, -13.8301620, 11.6761665, -25.5063248, 25.5063248
2: -23.0391483, 7.3856821, -23.0391483, 7.3856821, -30.4248314, 30.4248314
3: -20.5315437, 9.2545481, -20.5315437, 9.2545481, -29.7860909, 29.7860909
4: -20.7801514, 12.5645094, -20.7801514, 12.5645094, -33.3446579, 33.3446579
5: -15.4340830, 13.3476000, -15.4340830, 13.3476000, -28.7816830, 28.7816811
6: -16.4678898, 14.0741301, -16.4678898, 14.0741301, -30.5420189, 30.5420189
7: -18.6089172, 13.5067215, -18.6089172, 13.5067215, -32.1156387, 32.1156387
8: -21.4197464, 12.1960192, -21.4197464, 12.1960192, -33.6157608, 33.6157608
9: -14.8818188, 17.5486946, -14.8818188, 17.5486946, -32.4305115, 32.4305115

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1922204, upper bound: 27.1930815
time: 4.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
time: 4.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.7334938, 11.9369030, -17.2610950, 13.1040525, -28.8375435, 29.1979961
1: -12.5583439, 10.6374521, -13.8069859, 11.6735859, -24.2319298, 24.4444389
2: -20.9735107, 6.7049789, -22.8669186, 7.5812402, -28.5547504, 29.5718975
3: -18.6482773, 8.4368429, -20.4856758, 9.2972507, -27.9455261, 28.9225178
4: -18.9046631, 11.4424171, -20.6814804, 12.5680256, -31.4726868, 32.1238976
5: -14.0293102, 12.1520329, -15.4383678, 13.3064117, -27.3357201, 27.5903988
6: -14.9687243, 12.8095026, -16.4454594, 14.0625916, -29.0313148, 29.2549629
7: -16.9219570, 12.2906389, -18.5516663, 13.5539436, -30.4759007, 30.8423042
8: -19.4781837, 11.1107807, -21.3546505, 12.2006378, -31.6788177, 32.4654312
9: -13.5383987, 15.9693871, -14.8719521, 17.4647160, -31.0031128, 30.8413334

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1929309, upper bound: 27.1933619
time: 6.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1916469, upper bound: 27.1911743
time: 4.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.9330330, 12.0862713, -18.3878765, 13.9460621, -29.8790932, 30.4741478
1: -12.7195988, 10.7698059, -14.7204437, 12.4264927, -25.1460915, 25.4902496
2: -21.2332172, 6.7949381, -24.3400421, 8.0835571, -29.3167744, 31.1349773
3: -18.8858185, 8.5411921, -21.8287697, 9.8871880, -28.7730064, 30.3699608
4: -19.1408081, 11.5842705, -22.0118637, 13.3677769, -32.5085831, 33.5961342
5: -14.2070770, 12.3037930, -16.4490891, 14.1642952, -28.3713722, 28.7528801
6: -15.1586037, 12.9699535, -17.5317516, 14.9738636, -30.1324596, 30.5017052
7: -17.1362457, 12.4460955, -19.7593803, 14.4294586, -31.5657043, 32.2054749
8: -19.7245827, 11.2484522, -22.7565155, 12.9829140, -32.7074966, 34.0049667
9: -13.7096291, 16.1677799, -15.8402109, 18.5892372, -32.2988663, 32.0079842

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1929552, upper bound: 27.1933706
time: 3.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1916671, upper bound: 27.1911860
time: 4.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.3086624, 13.1110954, -17.3664093, 13.1840649, -30.4927273, 30.4775009
1: -13.8301620, 11.6761665, -13.8961716, 11.7444916, -25.5746536, 25.5723324
2: -23.0391483, 7.3856821, -23.0205383, 7.6025352, -30.6416836, 30.4062157
3: -20.5315437, 9.2545481, -20.5968895, 9.3359022, -29.8674469, 29.8514366
4: -20.7801514, 12.5645094, -20.8112106, 12.6294336, -33.4095802, 33.3757172
5: -15.4340830, 13.3476000, -15.5342627, 13.3937263, -28.8278084, 28.8818626
6: -16.4678898, 14.0741301, -16.5527077, 14.1519051, -30.6197929, 30.6268387
7: -18.6089172, 13.5067215, -18.6795216, 13.6284981, -32.2374153, 32.1862411
8: -21.4197464, 12.1960192, -21.5059509, 12.2588511, -33.6785965, 33.7019691
9: -14.8818188, 17.5486946, -14.9652519, 17.5726089, -32.4544296, 32.5139465

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1932004, upper bound: 27.1935331
time: 5.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918235, upper bound: 27.1913042
time: 18.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.3086624, 13.1110954, -18.7227573, 14.2010355, -31.5096931, 31.8338470
1: -13.8301620, 11.6761665, -14.9935465, 12.6504822, -26.4806404, 26.6697083
2: -23.0391483, 7.3856821, -24.7911720, 8.2266684, -31.2658157, 32.1768532
3: -20.5315437, 9.2545481, -22.2230797, 10.0535164, -30.5850601, 31.4776268
4: -20.7801514, 12.5645094, -22.4221344, 13.6065416, -34.3866882, 34.9866447
5: -15.4340830, 13.3476000, -16.7527390, 14.4238949, -29.8579750, 30.1003380
6: -16.4678898, 14.0741301, -17.8588409, 15.2488451, -31.7167320, 31.9329720
7: -18.6089172, 13.5067215, -20.1265373, 14.6859989, -33.2949142, 33.6332550
8: -21.4197464, 12.1960192, -23.1871510, 13.2136288, -34.6333771, 35.3831711
9: -14.8818188, 17.5486946, -16.1301155, 18.9302635, -33.8120804, 33.6788101

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1932004, upper bound: 27.1935331
time: 5.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918235, upper bound: 27.1913042
time: 5.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.3664093, 13.1840649, -17.0256939, 12.9024696, -30.2688789, 30.2097588
1: -13.8961716, 11.7444916, -13.6007490, 11.4886112, -25.3847809, 25.3452415
2: -23.0205383, 7.6025352, -22.6536369, 7.2771072, -30.2976418, 30.2561684
3: -20.5968895, 9.3359022, -20.1921387, 9.1147385, -29.7116280, 29.5280399
4: -20.8112106, 12.6294336, -20.4353695, 12.3647251, -33.1759338, 33.0648003
5: -15.5342627, 13.3937263, -15.1843166, 13.1317902, -28.6660500, 28.5780430
6: -16.5527077, 14.1519051, -16.2016716, 13.8452625, -30.3979664, 30.3535767
7: -18.6795216, 13.6284981, -18.3026276, 13.2946835, -31.9742050, 31.9311256
8: -21.5059509, 12.2588511, -21.0666389, 12.0034809, -33.5094299, 33.3254852
9: -14.9652519, 17.5726089, -14.6406431, 17.2622051, -32.2274551, 32.2132492

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1933734, upper bound: 27.1929838
time: 5.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1933787, upper bound: 27.1930035
time: 9.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.1882019, 13.0504761, -20.0851307, 15.2364426, -32.4246407, 33.1356049
1: -13.7508678, 11.6253681, -16.1005745, 13.5622196, -27.3130875, 27.7259407
2: -22.7874794, 7.5217032, -26.5709457, 8.8136921, -31.6011715, 34.0926437
3: -20.3838120, 9.2425919, -23.8911934, 10.7784395, -31.1622505, 33.1337852
4: -20.6001301, 12.5014381, -24.0719776, 14.5925684, -35.1926994, 36.5734177
5: -15.3735676, 13.2586212, -18.0007496, 15.4684067, -30.8419743, 31.2593708
6: -16.3819885, 14.0076504, -19.1505184, 16.3435268, -32.7255135, 33.1581650
7: -18.4888077, 13.4896975, -21.5906277, 15.7344074, -34.2232132, 35.0803261
8: -21.2850857, 12.1344404, -24.8927250, 14.1446533, -35.4297333, 37.0271645
9: -14.8122978, 17.3940163, -17.3022385, 20.3124542, -35.1247482, 34.6962547

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1911429, upper bound: 27.1916716
time: 4.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1911567, upper bound: 27.1916883
time: 6.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18.7227573, 14.2010355, -15.9330330, 12.0862713, -30.8090267, 30.1340637
1: -14.9935465, 12.6504822, -12.7195988, 10.7698059, -25.7633514, 25.3700771
2: -24.7911720, 8.2266684, -21.2332172, 6.7949381, -31.5861053, 29.4598827
3: -22.2230797, 10.0535164, -18.8858185, 8.5411921, -30.7642670, 28.9393349
4: -22.4221344, 13.6065416, -19.1408081, 11.5842705, -34.0064049, 32.7473412
5: -16.7527390, 14.4238949, -14.2070770, 12.3037930, -29.0565300, 28.6309719
6: -17.8588409, 15.2488451, -15.1586037, 12.9699535, -30.8287888, 30.4074440
7: -20.1265373, 14.6859989, -17.1362457, 12.4460955, -32.5726280, 31.8222446
8: -23.1871510, 13.2136288, -19.7245827, 11.2484522, -34.4356041, 32.9382095
9: -16.1301155, 18.9302635, -13.7096291, 16.1677799, -32.2978935, 32.6398926

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1967502, upper bound: 27.1970994
time: 5.13 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1970284, upper bound: 27.1975091
time: 6.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18.7227573, 14.2010355, -17.3086624, 13.1110954, -31.8338470, 31.5096931
1: -14.9935465, 12.6504822, -13.8301620, 11.6761665, -26.6697083, 26.4806404
2: -24.7911720, 8.2266684, -23.0391483, 7.3856821, -32.1768532, 31.2658157
3: -22.2230797, 10.0535164, -20.5315437, 9.2545481, -31.4776268, 30.5850601
4: -22.4221344, 13.6065416, -20.7801514, 12.5645094, -34.9866409, 34.3866920
5: -16.7527390, 14.4238949, -15.4340830, 13.3476000, -30.1003380, 29.8579750
6: -17.8588409, 15.2488451, -16.4678898, 14.0741301, -31.9329720, 31.7167320
7: -20.1265373, 14.6859989, -18.6089172, 13.5067215, -33.6332512, 33.2949142
8: -23.1871510, 13.2136288, -21.4197464, 12.1960192, -35.3831635, 34.6333771
9: -16.1301155, 18.9302635, -14.8818188, 17.5486946, -33.6788101, 33.8120804

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1967502, upper bound: 27.1971005
time: 4.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1970284, upper bound: 27.1975217
time: 4.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.1490364, 13.0201950, -17.2610950, 13.1040525, -30.2530899, 30.2812901
1: -13.7198524, 11.5969505, -13.8069859, 11.6735859, -25.3934383, 25.4039345
2: -22.7407188, 7.4926767, -22.8669186, 7.5812402, -30.3219585, 30.3595963
3: -20.3404503, 9.2200508, -20.4856758, 9.2972507, -29.6377010, 29.7057247
4: -20.5557365, 12.4717350, -20.6814804, 12.5680256, -33.1237640, 33.1532097
5: -15.3373556, 13.2286177, -15.4383678, 13.3064117, -28.6437683, 28.6669846
6: -16.3399467, 13.9748487, -16.4454594, 14.0625916, -30.4025307, 30.4203053
7: -18.4472752, 13.4563103, -18.5516663, 13.5539436, -32.0012207, 32.0079765
8: -21.2324257, 12.1037235, -21.3546505, 12.2006378, -33.4330635, 33.4583740
9: -14.7766094, 17.3577023, -14.8719521, 17.4647160, -32.2413254, 32.2296524

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941111, upper bound: 27.1950073
time: 4.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927694, upper bound: 27.1928087
time: 12.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17.3664093, 13.1840649, -18.3878765, 13.9460621, -31.3124714, 31.5719414
1: -13.8961716, 11.7444916, -14.7204437, 12.4264927, -26.3226643, 26.4649315
2: -23.0205383, 7.6025352, -24.3400421, 8.0835571, -31.1040859, 31.9425755
3: -20.5968895, 9.3359022, -21.8287697, 9.8871880, -30.4840775, 31.1646729
4: -20.8112106, 12.6294336, -22.0118637, 13.3677769, -34.1789856, 34.6412926
5: -15.5342627, 13.3937263, -16.4490891, 14.1642952, -29.6985512, 29.8428154
6: -16.5527077, 14.1519051, -17.5317516, 14.9738636, -31.5265713, 31.6836567
7: -18.6795216, 13.6284981, -19.7593803, 14.4294586, -33.1089783, 33.3878784
8: -21.5059509, 12.2588511, -22.7565155, 12.9829140, -34.4888611, 35.0153656
9: -14.9652519, 17.5726089, -15.8402109, 18.5892372, -33.5544853, 33.4128151

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1984114, upper bound: 27.1983592
time: 3.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1984114, upper bound: 27.1987578
time: 7.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18.7227573, 14.2010355, -17.3664093, 13.1840649, -31.9068203, 31.5674400
1: -14.9935465, 12.6504822, -13.8961716, 11.7444916, -26.7380371, 26.5466499
2: -24.7911720, 8.2266684, -23.0205383, 7.6025352, -32.3937073, 31.2471962
3: -22.2230797, 10.0535164, -20.5968895, 9.3359022, -31.5589771, 30.6504059
4: -22.4221344, 13.6065416, -20.8112106, 12.6294336, -35.0515671, 34.4177475
5: -16.7527390, 14.4238949, -15.5342627, 13.3937263, -30.1464653, 29.9581566
6: -17.8588409, 15.2488451, -16.5527077, 14.1519051, -32.0107460, 31.8015518
7: -20.1265373, 14.6859989, -18.6795216, 13.6284981, -33.7550316, 33.3655205
8: -23.1871510, 13.2136288, -21.5059509, 12.2588511, -35.4460030, 34.7195778
9: -16.1301155, 18.9302635, -14.9652519, 17.5726089, -33.7027245, 33.8955154

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1984317, upper bound: 27.1983263
time: 4.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987542, upper bound: 27.1987360
time: 4.25 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18.7227573, 14.2010355, -18.7227573, 14.2010355, -32.9237862, 32.9237862
1: -14.9935465, 12.6504822, -14.9935465, 12.6504822, -27.6440239, 27.6440239
2: -24.7911720, 8.2266684, -24.7911720, 8.2266684, -33.0178375, 33.0178375
3: -22.2230797, 10.0535164, -22.2230797, 10.0535164, -32.2765961, 32.2765961
4: -22.4221344, 13.6065416, -22.4221344, 13.6065416, -36.0286751, 36.0286751
5: -16.7527390, 14.4238949, -16.7527390, 14.4238949, -31.1766338, 31.1766338
6: -17.8588409, 15.2488451, -17.8588409, 15.2488451, -33.1076851, 33.1076851
7: -20.1265373, 14.6859989, -20.1265373, 14.6859989, -34.8125381, 34.8125381
8: -23.1871510, 13.2136288, -23.1871510, 13.2136288, -36.4007721, 36.4007797
9: -16.1301155, 18.9302635, -16.1301155, 18.9302635, -35.0603790, 35.0603790

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1984317, upper bound: 27.1983695
time: 26.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987542, upper bound: 27.1987823
time: 3.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.11 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1908211, upper bound: 27.1908548
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1908211, upper bound: 27.1908548
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1907258, upper bound: 27.1907534
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1907309, upper bound: 27.1907566
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1922204, upper bound: 27.1930815
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1922204, upper bound: 27.1930815
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1929309, upper bound: 27.1933619
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1916469, upper bound: 27.1911743
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1929552, upper bound: 27.1933706
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1916671, upper bound: 27.1911860
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1932004, upper bound: 27.1935331
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1918235, upper bound: 27.1913042
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1932004, upper bound: 27.1935331
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1918235, upper bound: 27.1913042
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1933734, upper bound: 27.1929838
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1933787, upper bound: 27.1930035
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1911429, upper bound: 27.1916716
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1911567, upper bound: 27.1916883
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1967502, upper bound: 27.1970994
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1970284, upper bound: 27.1975091
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1967502, upper bound: 27.1971005
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1970284, upper bound: 27.1975217
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1941111, upper bound: 27.1950073
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1927694, upper bound: 27.1928087
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1984114, upper bound: 27.1983592
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1984114, upper bound: 27.1987578
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1984317, upper bound: 27.1983263
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1987542, upper bound: 27.1987360
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1984317, upper bound: 27.1983695
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.11
Output dim: 2, lower bound: -27.1987542, upper bound: 27.1987823

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.3574076, 11.6566257, -17.0256939, 12.9024696, -28.2598763, 28.6823196
1: -12.2525272, 10.3881826, -13.6007490, 11.4886112, -23.7411385, 23.9889259
2: -20.4775772, 6.5401058, -22.6536369, 7.2771072, -27.7546844, 29.1937428
3: -18.1955338, 8.2413778, -20.1921387, 9.1147385, -27.3102722, 28.4335155
4: -18.4576073, 11.1739063, -20.4353695, 12.3647251, -30.8223305, 31.6092758
5: -13.6931591, 11.8669300, -15.1843166, 13.1317902, -26.8249474, 27.0512447
6: -14.6109324, 12.5070057, -16.2016716, 13.8452625, -28.4561939, 28.7086773
7: -16.5187798, 11.9999771, -18.3026276, 13.2946835, -29.8134632, 30.3026047
8: -19.0148430, 10.8499260, -21.0666389, 12.0034809, -31.0183201, 31.9165611
9: -13.2173243, 15.5919466, -14.6406431, 17.2622051, -30.4795284, 30.2325897

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1929080, upper bound: 27.1920313
time: 24.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1929948, upper bound: 27.1920734
time: 4.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18.7843781, 14.2420721, -17.0256939, 12.9024696, -31.6868439, 31.2677650
1: -15.0454741, 12.6952553, -13.6007490, 11.4886112, -26.5340843, 26.2960052
2: -24.8805008, 8.2057915, -22.6536369, 7.2771072, -32.1576042, 30.8594284
3: -22.3479614, 10.0821114, -20.1921387, 9.1147385, -31.4626961, 30.2742500
4: -22.5347652, 13.6488514, -20.4353695, 12.3647251, -34.8994865, 34.0842209
5: -16.8013630, 14.4832687, -15.1843166, 13.1317902, -29.9331532, 29.6675854
6: -17.8882713, 15.2967196, -16.2016716, 13.8452625, -31.7335224, 31.4983902
7: -20.2085743, 14.7163591, -18.3026276, 13.2946835, -33.5032578, 33.0189819
8: -23.2770576, 13.2274342, -21.0666389, 12.0034809, -35.2805290, 34.2940712
9: -16.1914787, 18.9991436, -14.6406431, 17.2622051, -33.4536819, 33.6397858

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1929080, upper bound: 27.1920313
time: 8.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1929948, upper bound: 27.1920734
time: 5.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.1490974, 10.7614031, -19.8911648, 15.0886240, -29.2377205, 30.6525688
1: -11.2800426, 9.5917015, -15.9435101, 13.4313545, -24.7113914, 25.5352116
2: -18.8933029, 6.0351720, -26.3188591, 8.7201080, -27.6134109, 32.3540306
3: -16.7465401, 7.6175246, -23.6621857, 10.6749344, -27.4214745, 31.2797108
4: -17.0255928, 10.3237705, -23.8424225, 14.4519205, -31.4775085, 34.1661911
5: -12.6201134, 10.9493513, -17.8251991, 15.3201160, -27.9402275, 28.7745438
6: -13.4625664, 11.5356989, -18.9615574, 16.1864491, -29.6490078, 30.4972572
7: -15.2214336, 11.0725327, -21.3829060, 15.5817862, -30.8032188, 32.4554367
8: -17.5194016, 10.0245361, -24.6479969, 14.0067024, -31.5260963, 34.6725235
9: -12.1839523, 14.3858910, -17.1344547, 20.1187057, -32.3026581, 31.5203438

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907258, upper bound: 27.1907534
time: 7.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907258, upper bound: 27.1907534
time: 4.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15.1430397, 11.4959850, -20.0851307, 15.2364426, -30.3794823, 31.5811119
1: -12.0824032, 10.2471838, -16.1005745, 13.5622196, -25.6446228, 26.3477592
2: -20.2044868, 6.4424019, -26.5709457, 8.8136921, -29.0181789, 33.0133438
3: -17.9442291, 8.1294270, -23.8911934, 10.7784395, -28.7226677, 32.0206223
4: -18.2050343, 11.0233126, -24.0719776, 14.5925684, -32.7976036, 35.0952911
5: -13.5037336, 11.7036266, -18.0007496, 15.4684067, -28.9721413, 29.7043762
6: -14.4078579, 12.3347635, -19.1505184, 16.3435268, -30.7513809, 31.4852829
7: -16.2877998, 11.8330212, -21.5906277, 15.7344074, -32.0222092, 33.4236450
8: -18.7493286, 10.7043343, -24.8927250, 14.1446533, -32.8939819, 35.5970612
9: -13.0324879, 15.3814526, -17.3022385, 20.3124542, -33.3449402, 32.6836929

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907295, upper bound: 27.1907538
time: 8.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907295, upper bound: 27.1907566
time: 4.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.7387295, 12.6842022, -15.9330330, 12.0862713, -28.8250008, 28.6172333
1: -13.3673725, 11.2979517, -12.7195988, 10.7698059, -24.1371784, 24.0175495
2: -22.2926311, 7.1320944, -21.2332172, 6.7949381, -29.0875664, 28.3653107
3: -19.8491707, 8.9563723, -18.8858185, 8.5411921, -28.3903618, 27.8421898
4: -20.1033726, 12.1577168, -19.1408081, 11.5842705, -31.6876431, 31.2985115
5: -14.9219494, 12.9148817, -14.2070770, 12.3037930, -27.2257385, 27.1219559
6: -15.9244957, 13.6156082, -15.1586037, 12.9699535, -28.8944473, 28.7742119
7: -17.9982929, 13.0646448, -17.1362457, 12.4460955, -30.4443893, 30.2008877
8: -20.7167587, 11.8021202, -19.7245827, 11.2484522, -31.9652100, 31.5267029
9: -14.3949442, 16.9773674, -13.7096291, 16.1677799, -30.5627213, 30.6869965

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908548, upper bound: 27.1908211
time: 5.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908548, upper bound: 27.1908211
time: 3.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -19.7770653, 14.9981947, -15.7699070, 11.9646492, -31.7417145, 30.7680988
1: -15.8524733, 13.3526917, -12.5871677, 10.6615982, -26.5140705, 25.9398594
2: -26.1819725, 8.6439133, -21.0193863, 6.7231970, -32.9051704, 29.6632996
3: -23.5304089, 10.6069946, -18.6903877, 8.4563627, -31.9867687, 29.2973824
4: -23.7172775, 14.3625784, -18.9471169, 11.4680748, -35.1853523, 33.3096962
5: -17.7174339, 15.2357445, -14.0615606, 12.1798630, -29.8972969, 29.2973022
6: -18.8389397, 16.0951538, -15.0033865, 12.8387241, -31.6776638, 31.0985394
7: -21.2681923, 15.4869490, -16.9613152, 12.3198795, -33.5880699, 32.4482613
8: -24.5009499, 13.9132061, -19.5232105, 11.1358242, -35.6367683, 33.4364166
9: -17.0351868, 20.0054703, -13.5702267, 16.0045109, -33.0396957, 33.5756912

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907534, upper bound: 27.1907258
time: 5.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907566, upper bound: 27.1907309
time: 7.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -16.7387295, 12.6842022, -17.3086624, 13.1110954, -29.8498192, 29.9928627
1: -13.3673725, 11.2979517, -13.8301620, 11.6761665, -25.0435371, 25.1281128
2: -22.2926311, 7.1320944, -23.0391483, 7.3856821, -29.6783123, 30.1712418
3: -19.8491707, 8.9563723, -20.5315437, 9.2545481, -29.1037178, 29.4879150
4: -20.1033726, 12.1577168, -20.7801514, 12.5645094, -32.6678772, 32.9378624
5: -14.9219494, 12.9148817, -15.4340830, 13.3476000, -28.2695503, 28.3489609
6: -15.9244957, 13.6156082, -16.4678898, 14.0741301, -29.9986267, 30.0834980
7: -17.9982929, 13.0646448, -18.6089172, 13.5067215, -31.5050144, 31.6735611
8: -20.7167587, 11.8021202, -21.4197464, 12.1960192, -32.9127655, 33.2218666
9: -14.3949442, 16.9773674, -14.8818188, 17.5486946, -31.9436378, 31.8591862

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
time: 5.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
time: 3.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -19.7770653, 14.9981947, -17.1500111, 12.9921875, -32.7692490, 32.1482048
1: -15.8524733, 13.3526917, -13.7012568, 11.5707169, -27.4231911, 27.0539474
2: -26.1819725, 8.6439133, -22.8308411, 7.3154335, -33.4974060, 31.4747543
3: -23.5304089, 10.6069946, -20.3415756, 9.1717558, -32.7021637, 30.9485703
4: -23.7172775, 14.3625784, -20.5914040, 12.4512730, -36.1685486, 34.9539833
5: -17.7174339, 15.2357445, -15.2915154, 13.2269840, -30.9444180, 30.5272579
6: -18.8389397, 16.0951538, -16.3166008, 13.9463320, -32.7852707, 32.4117508
7: -21.2681923, 15.4869490, -18.4387360, 13.3837261, -34.6519165, 33.9256859
8: -24.5009499, 13.9132061, -21.2237206, 12.0864201, -36.5873680, 35.1369247
9: -17.0351868, 20.0054703, -14.7461920, 17.3895397, -34.4247246, 34.7516632

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907262, upper bound: 27.1907244
time: 4.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1907283, upper bound: 27.1907283
time: 6.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.1586838, 11.5078516, -17.2610950, 13.1040525, -28.2627296, 28.7689457
1: -12.0922747, 10.2565079, -13.8069859, 11.6735859, -23.7658596, 24.0634918
2: -20.2175827, 6.4519105, -22.8669186, 7.5812402, -27.7988224, 29.3188286
3: -17.9581032, 8.1378345, -20.4856758, 9.2972507, -27.2553539, 28.6235104
4: -18.2215919, 11.0331726, -20.6814804, 12.5680256, -30.7896118, 31.7146492
5: -13.5161085, 11.7156048, -15.4383678, 13.3064117, -26.8225212, 27.1539669
6: -14.4222374, 12.3469305, -16.4454594, 14.0625916, -28.4848289, 28.7923889
7: -16.3049622, 11.8456612, -18.5516663, 13.5539436, -29.8589058, 30.3973255
8: -18.7687492, 10.7132931, -21.3546505, 12.2006378, -30.9693832, 32.0679436
9: -13.0467854, 15.3941135, -14.8719521, 17.4647160, -30.5115013, 30.2660580

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1916469, upper bound: 27.1911743
time: 62.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1916469, upper bound: 27.1911743
time: 3.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 67.50 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1929080, upper bound: 27.1920313
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1929948, upper bound: 27.1920734
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1929080, upper bound: 27.1920313
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1929948, upper bound: 27.1920734
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907258, upper bound: 27.1907534
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907258, upper bound: 27.1907534
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907295, upper bound: 27.1907538
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907295, upper bound: 27.1907566
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1908548, upper bound: 27.1908211
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1908548, upper bound: 27.1908211
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907534, upper bound: 27.1907258
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907566, upper bound: 27.1907309
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1908172, upper bound: 27.1908172
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907262, upper bound: 27.1907244
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1907283, upper bound: 27.1907283
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1916469, upper bound: 27.1911743
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 67.50
Output dim: 2, lower bound: -27.1916469, upper bound: 27.1911743
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1916469, upper bound: 27.1911743
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1929552, upper bound: 27.1933706
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1916671, upper bound: 27.1911860
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1932004, upper bound: 27.1935331
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1918235, upper bound: 27.1913042
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1932004, upper bound: 27.1935331
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1918235, upper bound: 27.1913042
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1933734, upper bound: 27.1929838
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1933787, upper bound: 27.1930035
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1911429, upper bound: 27.1916716
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1911567, upper bound: 27.1916883
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1967502, upper bound: 27.1970994
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1970284, upper bound: 27.1975091
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1967502, upper bound: 27.1971005
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1970284, upper bound: 27.1975217
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1941111, upper bound: 27.1950073
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1927694, upper bound: 27.1928087
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1984114, upper bound: 27.1983592
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1984114, upper bound: 27.1987578
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1984317, upper bound: 27.1983263
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1987542, upper bound: 27.1987360
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1984317, upper bound: 27.1983695
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 67.50
Output dim: 2, lower bound: -27.1987542, upper bound: 27.1987823
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=36.862388610839844
rel_dist={2: [-27.200554238179546, 27.20055423068736]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
time: 5.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716
time: 5.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.10
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969081
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.10
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -19.5840302, 14.8673496, -32.6044312, 33.0222969
1: -14.1783390, 11.9624004, -15.7014399, 13.2516689, -27.4300079, 27.6638412
2: -23.5818024, 7.6063967, -25.8618412, 8.7477245, -32.3295212, 33.4682388
3: -21.0418549, 9.4901848, -23.2685165, 10.5544968, -31.5963516, 32.7587013
4: -21.2769165, 12.8763294, -23.4271584, 14.2579031, -35.5348206, 36.3034897
5: -15.8263702, 13.6713247, -17.5454559, 15.0845242, -30.9108944, 31.2167816
6: -16.8846321, 14.4183712, -18.7129440, 15.9619741, -32.8466072, 33.1313171
7: -19.0637054, 13.8504381, -21.0339527, 15.3976727, -34.4613762, 34.8843880
8: -21.9471779, 12.5008354, -24.2660751, 13.8591270, -35.8062973, 36.7669067
9: -15.2502155, 17.9744301, -16.8939228, 19.7750587, -35.0252762, 34.8683472

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969037
time: 4.01 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969065
time: 4.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -20.0912285, 15.2757206, -34.5570946, 34.7240829
1: -15.4504633, 13.0406036, -16.1221123, 13.6216984, -29.0721626, 29.1627159
2: -25.4806252, 8.5678139, -26.4521561, 9.1441746, -34.6247940, 35.0199623
3: -22.9028149, 10.3804493, -23.8861313, 10.8818111, -33.7846260, 34.2665787
4: -23.0672989, 14.0238628, -24.0047455, 14.6549339, -37.7222328, 38.0286064
5: -17.2664795, 14.8502493, -18.0368500, 15.4769497, -32.7434311, 32.8871002
6: -18.4147415, 15.7105303, -19.2292881, 16.3985214, -34.8132591, 34.9398193
7: -20.7114697, 15.1501713, -21.5708580, 15.8566313, -36.5681000, 36.7210312
8: -23.8846569, 13.6342249, -24.8998566, 14.2604790, -38.1451340, 38.5340805
9: -16.6258526, 19.4761505, -17.3630238, 20.2495880, -36.8754425, 36.8391685

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987596, upper bound: 27.1987593
time: 8.25 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716
time: 4.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.21 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969037
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 2, lower bound: -27.1972990, upper bound: 27.1969065
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 2, lower bound: -27.1987596, upper bound: 27.1987593
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987716

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.3370628, 13.1330891, -17.6276932, 13.3823795, -30.7194405, 30.7607822
1: -13.8530216, 11.6943493, -14.1119061, 11.9265022, -25.7795238, 25.8062515
2: -23.0674133, 7.4033875, -23.3507347, 7.7548943, -30.8223076, 30.7541199
3: -20.5658150, 9.2729082, -20.9079361, 9.4832172, -30.0490322, 30.1808434
4: -20.8094063, 12.5843725, -21.1190586, 12.8308964, -33.6403046, 33.7034302
5: -15.4613705, 13.3685026, -15.7731667, 13.5933065, -29.0546761, 29.1416702
6: -16.4964905, 14.0960989, -16.8093491, 14.3677006, -30.8641891, 30.9054413
7: -18.6369781, 13.5307159, -18.9578648, 13.8396053, -32.4765854, 32.4885788
8: -21.4527378, 12.2158527, -21.8363914, 12.4509439, -33.9036789, 34.0522461
9: -14.9058094, 17.5759335, -15.1942959, 17.8303185, -32.7361298, 32.7702255

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1922682, upper bound: 27.1924932
time: 5.89 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1914562, upper bound: 27.1910880
time: 15.64 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.5927582, 13.3273621, -19.0028114, 14.4149561, -32.0077057, 32.3301735
1: -14.0613518, 11.8653307, -15.2250328, 12.8454943, -26.9068451, 27.0903625
2: -23.3992157, 7.5290685, -25.1455765, 8.3917503, -31.7909660, 32.6746445
3: -20.8711929, 9.4107084, -22.5602036, 10.2133112, -31.0845032, 31.9709110
4: -21.1102295, 12.7700186, -22.7534103, 13.8232403, -34.9334717, 35.5234299
5: -15.6937466, 13.5625000, -17.0077801, 14.6391811, -30.3329277, 30.5702782
6: -16.7422428, 14.3019047, -18.1344185, 15.4804792, -32.2227211, 32.4363251
7: -18.9111023, 13.7342377, -20.4248276, 14.9128685, -33.8239708, 34.1590652
8: -21.7673512, 12.3959579, -23.5417595, 13.4206448, -35.1879959, 35.9377174
9: -15.1254759, 17.8310757, -16.3767147, 19.2071800, -34.3326569, 34.2077904

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1922428, upper bound: 27.1924823
time: 6.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1914297, upper bound: 27.1910797
time: 5.56 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -18.7854786, 14.2485313, -18.0481148, 13.7090950, -32.4945679, 32.2966423
1: -15.0435600, 12.6955757, -14.4594097, 12.2307062, -27.2742653, 27.1549854
2: -24.8579922, 8.2737198, -23.8371716, 8.0769901, -32.9349747, 32.1108856
3: -22.3003845, 10.0943928, -21.4192886, 9.7487450, -32.0491295, 31.5136795
4: -22.4895763, 13.6540966, -21.5930634, 13.1559553, -35.6455307, 35.2471619
5: -16.8097725, 14.4704704, -16.1716652, 13.9128895, -30.7226620, 30.6421280
6: -17.9238014, 15.3011847, -17.2357769, 14.7252369, -32.6490364, 32.5369606
7: -20.1883736, 14.7397346, -19.4009266, 14.2143097, -34.4026833, 34.1406555
8: -23.2649822, 13.2635117, -22.3658028, 12.7738342, -36.0388107, 35.6293106
9: -16.1866035, 18.9885826, -15.5786438, 18.2212715, -34.4078751, 34.5672226

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1983060, upper bound: 27.1982240
time: 5.62 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1986913, upper bound: 27.1986800
time: 8.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -19.0930195, 14.4850512, -19.4342861, 14.7581120, -33.8511276, 33.9193306
1: -15.2954569, 12.9083900, -15.5850296, 13.1613369, -28.4567947, 28.4934196
2: -25.2478142, 8.4515581, -25.6462631, 8.7288589, -33.9766693, 34.0978165
3: -22.6728954, 10.2697258, -23.0882168, 10.4903326, -33.1632271, 33.3579407
4: -22.8479252, 13.8824434, -23.2434616, 14.1596613, -37.0075874, 37.1259003
5: -17.0913181, 14.7049208, -17.4215775, 14.9717159, -32.0630341, 32.1264992
6: -18.2270794, 15.5543003, -18.5728054, 15.8502941, -34.0773697, 34.1271057
7: -20.5140228, 14.9919472, -20.8795033, 15.3032761, -35.8172989, 35.8714485
8: -23.6493340, 13.4916534, -24.0835838, 13.7568226, -37.4061584, 37.5752373
9: -16.4574127, 19.2913055, -16.7756424, 19.6099205, -36.0673294, 36.0669479

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1983376, upper bound: 27.1982713
time: 14.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987290, upper bound: 27.1987290
time: 5.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1922682, upper bound: 27.1924932
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1914562, upper bound: 27.1910880
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1922428, upper bound: 27.1924823
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1914297, upper bound: 27.1910797
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1983060, upper bound: 27.1982240
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1986913, upper bound: 27.1986800
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1983376, upper bound: 27.1982713
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.65
Output dim: 2, lower bound: -27.1987290, upper bound: 27.1987290

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.7611084, 12.7015238, -17.4635105, 13.2587433, -30.0198517, 30.1650352
1: -13.3855009, 11.3122377, -13.9779024, 11.8166933, -25.2021942, 25.2901402
2: -22.3123016, 7.1470265, -23.1363297, 7.6785393, -29.9908409, 30.2833557
3: -19.8758621, 8.9715996, -20.7107811, 9.3963175, -29.2721786, 29.6823807
4: -20.1252022, 12.1731739, -20.9243851, 12.7126522, -32.8378525, 33.0975533
5: -14.9436359, 12.9313087, -15.6247940, 13.4686680, -28.4123039, 28.5561028
6: -15.9472513, 13.6328230, -16.6513863, 14.2345819, -30.1818275, 30.2842102
7: -18.0196018, 13.0838747, -18.7823830, 13.7111149, -31.7307167, 31.8662567
8: -20.7419281, 11.8177338, -21.6327553, 12.3358212, -33.0777512, 33.4504890
9: -14.4135876, 16.9984055, -15.0529184, 17.6658764, -32.0794601, 32.0513229

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920667, upper bound: 27.1923440
time: 6.18 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920848, upper bound: 27.1923600
time: 5.15 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -19.7926025, 15.0098991, -17.1615486, 13.0319195, -32.8245239, 32.1714478
1: -15.8650379, 13.3626194, -13.7317009, 11.6145725, -27.4796104, 27.0943203
2: -26.1953201, 8.6536350, -22.7421150, 7.5410781, -33.7363930, 31.3957500
3: -23.5491657, 10.6173306, -20.3493538, 9.2377615, -32.7869186, 30.9666843
4: -23.7330208, 14.3734350, -20.5664139, 12.4954166, -36.2284393, 34.9398499
5: -17.7322464, 15.2470036, -15.3522120, 13.2394705, -30.9717178, 30.5992050
6: -18.8550873, 16.1071377, -16.3611298, 13.9896927, -32.8447762, 32.4682693
7: -21.2829590, 15.5005894, -18.4591484, 13.4756565, -34.7586136, 33.9597359
8: -24.5181770, 13.9240952, -21.2576618, 12.1251736, -36.6433487, 35.1817513
9: -17.0485058, 20.0201340, -14.7933674, 17.3633423, -34.4118385, 34.8134995

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912846, upper bound: 27.1909368
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912946, upper bound: 27.1909477
time: 6.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -17.0233688, 12.9005690, -18.8389835, 14.2913227, -31.3146915, 31.7395496
1: -13.5988941, 11.4872417, -15.0911036, 12.7355528, -26.3344460, 26.5783463
2: -22.6535301, 7.2747264, -24.9321651, 8.3147659, -30.9682961, 32.2068901
3: -20.1892662, 9.1124811, -22.3620949, 10.1260185, -30.3152847, 31.4745750
4: -20.4339523, 12.3632784, -22.5590744, 13.7051744, -34.1391258, 34.9223518
5: -15.1817360, 13.1301622, -16.8597755, 14.5146275, -29.6963615, 29.9899368
6: -16.1990471, 13.8436117, -17.9767685, 15.3472919, -31.5463390, 31.8203754
7: -18.3009148, 13.2923727, -20.2498150, 14.7846193, -33.0855331, 33.5421829
8: -21.0644188, 12.0019073, -23.3385391, 13.3053226, -34.3697281, 35.3404465
9: -14.6387815, 17.2602444, -16.2352409, 19.0429306, -33.6817131, 33.4954834

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920447, upper bound: 27.1923309
time: 5.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1920548, upper bound: 27.1923435
time: 6.27 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.0843010, 15.2357140, -18.5362091, 14.0626316, -34.1469345, 33.7719231
1: -16.1000633, 13.5616837, -14.8433132, 12.5322275, -28.6322899, 28.4049950
2: -26.5721188, 8.8119392, -24.5363045, 8.1734180, -34.7455330, 33.3482437
3: -23.8904972, 10.7773151, -21.9966831, 9.9649811, -33.8554764, 32.7739944
4: -24.0719891, 14.5917912, -22.1993427, 13.4865837, -37.5585709, 36.7911339
5: -17.9997292, 15.4680061, -16.5862865, 14.2837315, -32.2834549, 32.0542908
6: -19.1487217, 16.3430138, -17.6854897, 15.1008635, -34.2495804, 34.0285034
7: -21.5906334, 15.7333527, -19.9259377, 14.5476456, -36.1382751, 35.6592827
8: -24.8916893, 14.1434441, -22.9623375, 13.0920734, -37.9837646, 37.1057816
9: -17.3014793, 20.3118572, -15.9739494, 18.7387543, -36.0402298, 36.2858047

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
time: 7.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
time: 6.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.9454861, 12.8668261, -17.3988724, 13.2133541, -30.1588402, 30.2656975
1: -13.5515566, 11.4587059, -13.9286137, 11.7878408, -25.3393974, 25.3873196
2: -22.4708424, 7.4094872, -22.9998798, 7.7392931, -30.2101326, 30.4093590
3: -20.1069412, 9.1206455, -20.6440544, 9.3951778, -29.5021191, 29.7646999
4: -20.3167496, 12.3378563, -20.8253002, 12.6819410, -32.9986877, 33.1631546
5: -15.1531582, 13.0687008, -15.5798740, 13.4151173, -28.5682716, 28.6485710
6: -16.1405029, 13.8068848, -16.5962906, 14.1920795, -30.3325825, 30.4031715
7: -18.2180462, 13.2977200, -18.7080917, 13.6946821, -31.9127274, 32.0058136
8: -20.9678802, 11.9746151, -21.5462704, 12.3052216, -33.2731018, 33.5208855
9: -14.5976534, 17.1575851, -15.0114813, 17.5755882, -32.1732407, 32.1690674

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941215, upper bound: 27.1935276
time: 4.44 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
time: 8.80 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18.0727615, 13.7078123, -17.7779789, 13.5019989, -31.5747566, 31.4857903
1: -14.4662971, 12.2095051, -14.2388258, 12.0458632, -26.5121574, 26.4483299
2: -23.9460373, 7.9026222, -23.4901066, 7.9343195, -31.8803558, 31.3927288
3: -21.4527779, 9.7081995, -21.0959034, 9.6011181, -31.0538902, 30.8041000
4: -21.6499386, 13.1350212, -21.2730961, 12.9585218, -34.6084595, 34.4081192
5: -16.1629543, 13.9270782, -15.9248333, 13.7055550, -29.8685093, 29.8519058
6: -17.2218952, 14.7180271, -16.9685650, 14.5027180, -31.7246132, 31.6865921
7: -19.4276237, 14.1729460, -19.1124020, 13.9975815, -33.4252052, 33.2853432
8: -22.3653793, 12.7510519, -22.0237732, 12.5780878, -34.9434662, 34.7748184
9: -15.5645599, 18.2840977, -15.3417206, 17.9526691, -33.5172272, 33.6258163

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941593, upper bound: 27.1935760
time: 7.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
time: 5.24 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.2527294, 13.0983181, -18.7897644, 14.2577639, -31.5104866, 31.8880806
1: -13.8006544, 11.6674900, -15.0551748, 12.7168999, -26.5175533, 26.7226639
2: -22.8606625, 7.5734234, -24.8164902, 8.3835373, -31.2441998, 32.3899117
3: -20.4755478, 9.2910318, -22.3137760, 10.1348743, -30.6104221, 31.6048088
4: -20.6735802, 12.5619392, -22.4773674, 13.6854219, -34.3590012, 35.0393066
5: -15.4310980, 13.3005295, -16.8287430, 14.4734278, -29.9045258, 30.1292725
6: -16.4368877, 14.0558109, -17.9373150, 15.3161306, -31.7530174, 31.9931240
7: -18.5442238, 13.5462551, -20.1908684, 14.7822590, -33.3264847, 33.7371216
8: -21.3451881, 12.1941366, -23.2707291, 13.2857018, -34.6308899, 35.4648590
9: -14.8643236, 17.4581394, -16.2072296, 18.9668121, -33.8311348, 33.6653557

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941304, upper bound: 27.1935494
time: 5.93 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
time: 4.27 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18.3778687, 13.9387369, -19.1625786, 14.5461378, -32.9240074, 33.1013145
1: -14.7128162, 12.4189720, -15.3619604, 12.9737701, -27.6865864, 27.7809334
2: -24.3315048, 8.0744801, -25.2989197, 8.5815573, -32.9130630, 33.3733978
3: -21.8161564, 9.8798647, -22.7616711, 10.3397675, -32.1559219, 32.6415367
4: -22.0021095, 13.3599682, -22.9199524, 13.9593830, -35.9614944, 36.2799225
5: -16.4401760, 14.1572275, -17.1705513, 14.7608681, -31.2010422, 31.3277779
6: -17.5210114, 14.9654789, -18.3045349, 15.6245995, -33.1456070, 33.2700081
7: -19.7501793, 14.4206543, -20.5894966, 15.0829058, -34.8330841, 35.0101509
8: -22.7443638, 12.9743996, -23.7414589, 13.5576429, -36.3020058, 36.7158585
9: -15.8308773, 18.5807457, -16.5352421, 19.3390675, -35.1699371, 35.1159897

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1941694, upper bound: 27.1935960
time: 6.96 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
time: 7.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.76 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1920667, upper bound: 27.1923440
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1920848, upper bound: 27.1923600
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1912846, upper bound: 27.1909368
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1912946, upper bound: 27.1909477
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1920447, upper bound: 27.1923309
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1920548, upper bound: 27.1923435
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1941215, upper bound: 27.1935276
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1941593, upper bound: 27.1935760
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1941304, upper bound: 27.1935494
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1941694, upper bound: 27.1935960
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.76
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.2179451, 12.2932873, -15.6298113, 11.8864088, -28.1043549, 27.9230938
1: -12.9455462, 10.9510746, -12.4897137, 10.5909882, -23.5365257, 23.4407864
2: -21.6050339, 6.8998470, -20.7523842, 6.8433084, -28.4483414, 27.6522312
3: -19.2290173, 8.6864014, -18.5304756, 8.4357395, -27.6647568, 27.2168770
4: -19.4810944, 11.7858524, -18.7578888, 11.4063644, -30.8874512, 30.5437412
5: -14.4570465, 12.5176792, -13.9771748, 12.0732813, -26.5303268, 26.4948540
6: -15.4295273, 13.1952209, -14.8856564, 12.7476292, -28.1771564, 28.0808773
7: -17.4357662, 12.6596889, -16.8171005, 12.2798710, -29.7156277, 29.4767876
8: -20.0704784, 11.4425259, -19.3510685, 11.0646057, -31.1350822, 30.7935944
9: -13.9471302, 16.4572392, -13.4754925, 15.8381977, -29.7853279, 29.9327316

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
time: 4.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1725830, upper bound: 27.1728371
time: 6.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16.5418148, 12.5360937, -16.7633343, 12.7316513, -29.2734604, 29.2994232
1: -13.2079363, 11.1662560, -13.4110098, 11.3436203, -24.5515556, 24.5772667
2: -22.0281219, 7.0460815, -22.2385750, 7.3291588, -29.3572807, 29.2846508
3: -19.6156807, 8.8558521, -19.8834877, 9.0235682, -28.6392441, 28.7393341
4: -19.8647614, 12.0166845, -20.0989819, 12.2082129, -32.0729752, 32.1156654
5: -14.7464275, 12.7638063, -14.9917889, 12.9358768, -27.6823006, 27.7555885
6: -15.7379799, 13.4557514, -15.9685974, 13.6642838, -29.4022598, 29.4243488
7: -17.7834473, 12.9121981, -18.0321121, 13.1566267, -30.9400749, 30.9443092
8: -20.4699364, 11.6668720, -20.7552090, 11.8434219, -32.3133583, 32.4220810
9: -14.2250366, 16.7799129, -14.4455919, 16.9747581, -31.1997929, 31.2254982

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
time: 4.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1727857, upper bound: 27.1729945
time: 4.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -19.2450123, 14.5961208, -15.3295527, 11.6617451, -30.9067497, 29.9256744
1: -15.4198303, 12.9963255, -12.2444410, 10.3909225, -25.8107471, 25.2407627
2: -25.4823742, 8.4027948, -20.3610001, 6.7097507, -32.1921234, 28.7637939
3: -22.8974152, 10.3273478, -18.1707001, 8.2786274, -31.1760426, 28.4980450
4: -23.0817757, 13.9818602, -18.4011192, 11.1912985, -34.2730751, 32.3829727
5: -17.2387733, 14.8279772, -13.7066708, 11.8452501, -29.0840225, 28.5346489
6: -18.3318958, 15.6644859, -14.5992002, 12.5039988, -30.8358879, 30.2636833
7: -20.6928139, 15.0706997, -16.4953308, 12.0465832, -32.7393951, 31.5660305
8: -23.8382378, 13.5459690, -18.9794693, 10.8569651, -34.6951981, 32.5254326
9: -16.5768471, 19.4735603, -13.2177391, 15.5371761, -32.1140213, 32.6912994

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746570, upper bound: 27.1750444
time: 27.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
time: 10.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -19.5698204, 14.8416986, -16.4579887, 12.5029869, -32.0728035, 31.2996826
1: -15.6843138, 13.2138824, -13.1619911, 11.1399097, -26.8242226, 26.3758698
2: -25.9062767, 8.5515594, -21.8390884, 7.1928687, -33.0991440, 30.3906479
3: -23.2841377, 10.4993858, -19.5180187, 8.8640156, -32.1481552, 30.0174046
4: -23.4680939, 14.2142210, -19.7367096, 11.9895000, -35.4575958, 33.9509239
5: -17.5315762, 15.0766478, -14.7168732, 12.7039700, -30.2355423, 29.7935219
6: -18.6423893, 15.9270267, -15.6771812, 13.4170904, -32.0594788, 31.6042042
7: -21.0427799, 15.3257761, -17.7051010, 12.9191923, -33.9619675, 33.0308762
8: -24.2415104, 13.7705793, -20.3778305, 11.6318026, -35.8733139, 34.1484108
9: -16.8565540, 19.7979851, -14.1840363, 16.6682777, -33.5248260, 33.9820137

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909453
time: 5.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909477
time: 4.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -16.4827156, 12.4924955, -16.9858341, 12.8967171, -29.3794327, 29.4783287
1: -13.1600828, 11.1268635, -13.5863924, 11.4882698, -24.6483498, 24.7132568
2: -21.9496689, 7.0267243, -22.5281563, 7.4377155, -29.3873825, 29.5548801
3: -19.5457077, 8.8272667, -20.1509781, 9.1429100, -28.6886101, 28.9782391
4: -19.7918701, 11.9770899, -20.3699417, 12.3765373, -32.1684074, 32.3470306
5: -14.6947384, 12.7177277, -15.1887407, 13.1009407, -27.7956791, 27.9064674
6: -15.6830111, 13.4073477, -16.1784515, 13.8394594, -29.5224686, 29.5858002
7: -17.7191010, 12.8691654, -18.2656956, 13.3297415, -31.0488358, 31.1348610
8: -20.3947258, 11.6285381, -21.0223770, 12.0031166, -32.3978424, 32.6509132
9: -14.1740932, 16.7206497, -14.6338034, 17.1963673, -31.3704586, 31.3544540

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
time: 7.19 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1722802, upper bound: 27.1726510
time: 7.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.8072109, 12.7373571, -18.1314869, 13.7519608, -30.5591717, 30.8688431
1: -13.4237556, 11.3432636, -14.5158672, 12.2519855, -25.6757412, 25.8591270
2: -22.3735237, 7.1748500, -24.0264778, 7.9421082, -30.3156319, 31.2013283
3: -19.9325829, 8.9981852, -21.5187569, 9.7410355, -29.6736183, 30.5169411
4: -20.1770668, 12.2089071, -21.7233887, 13.1888561, -33.3659210, 33.9322929
5: -14.9871206, 12.9650249, -16.2159653, 13.9729776, -28.9600983, 29.1809883
6: -15.9927444, 13.6689959, -17.2781448, 14.7655315, -30.7582760, 30.9471397
7: -18.0678749, 13.1230135, -19.4941692, 14.2198210, -32.2876968, 32.6171799
8: -20.7961826, 11.8531303, -22.4435921, 12.7945061, -33.5906830, 34.2967224
9: -14.4528065, 17.0447922, -15.6168671, 18.3410797, -32.7938843, 32.6616516

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755090, upper bound: 27.1763456
time: 4.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1725884, upper bound: 27.1728598
time: 4.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -19.5247192, 14.8095551, -16.6847229, 12.6712198, -32.1959381, 31.4942780
1: -15.6469765, 13.1842670, -13.3410034, 11.2871246, -26.9340992, 26.5252686
2: -25.8448658, 8.5424776, -22.1352348, 7.3016458, -33.1465073, 30.6777115
3: -23.2298965, 10.4788322, -19.7902756, 8.9847498, -32.2146416, 30.2691078
4: -23.4100189, 14.1861391, -20.0131493, 12.1605968, -35.5706062, 34.1992836
5: -17.4934692, 15.0403147, -14.9176588, 12.8726120, -30.3660774, 29.9579735
6: -18.6037083, 15.8899345, -15.8908596, 13.5957794, -32.1994820, 31.7807884
7: -20.9913826, 15.2933502, -17.9432507, 13.0949430, -34.0863228, 33.2365990
8: -24.1858101, 13.7457409, -20.6500072, 11.7940903, -35.9798889, 34.3957481
9: -16.8177071, 19.7527752, -14.3753357, 16.8951721, -33.7128754, 34.1281128

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
time: 5.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
time: 8.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -19.8576164, 15.0631142, -17.8284798, 13.5247116, -33.3823280, 32.8915939
1: -15.9170017, 13.4089794, -14.2692556, 12.0490637, -27.9660645, 27.6782341
2: -26.2785530, 8.7024059, -23.6308784, 7.8027625, -34.0813141, 32.3332825
3: -23.6231213, 10.6563911, -21.1564140, 9.5811052, -33.2042274, 31.8128052
4: -23.8037701, 14.4273643, -21.3647900, 12.9706621, -36.7744331, 35.7921486
5: -17.7946472, 15.2948818, -15.9428644, 13.7430964, -31.5377312, 31.2377415
6: -18.9276924, 16.1593914, -16.9878063, 14.5202427, -33.4479294, 33.1471977
7: -21.3479652, 15.5552454, -19.1698112, 13.9834003, -35.3313675, 34.7250557
8: -24.6052628, 13.9822197, -22.0679493, 12.5826683, -37.1879311, 36.0501633
9: -17.1053638, 20.0855179, -15.3565264, 18.0384254, -35.1437836, 35.4420433

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
time: 7.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
time: 6.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.7848148, 12.7463837, -16.7486172, 12.7244434, -29.5092583, 29.4949951
1: -13.4206581, 11.3514471, -13.3982906, 11.3526688, -24.7733250, 24.7497349
2: -22.2612381, 7.3360648, -22.1523628, 7.4361439, -29.6973801, 29.4884186
3: -19.9142666, 9.0360432, -19.8633099, 9.0499077, -28.9641743, 28.8993492
4: -20.1264935, 12.2226000, -20.0551453, 12.2138090, -32.3403015, 32.2777443
5: -15.0083179, 12.9470596, -14.9929352, 12.9211369, -27.9294548, 27.9399910
6: -15.9870701, 13.6770363, -15.9717302, 13.6653500, -29.6524181, 29.6487656
7: -18.0461292, 13.1722403, -18.0138855, 13.1859341, -31.2320633, 31.1861229
8: -20.7694855, 11.8628893, -20.7408447, 11.8494682, -32.6189537, 32.6037292
9: -14.4598036, 16.9970016, -14.4529314, 16.9249992, -31.3848019, 31.4499321

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
time: 9.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
time: 5.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16.4938164, 12.5281315, -20.7298088, 15.7321959, -32.2260132, 33.2579422
1: -13.1829309, 11.1569796, -16.6774101, 14.0822372, -27.2651672, 27.8343887
2: -21.8808537, 7.2055979, -27.1901531, 9.5459099, -31.4267635, 34.3957520
3: -19.5650787, 8.8835812, -24.6946297, 11.2803822, -30.8454609, 33.5782089
4: -19.7808762, 12.0142813, -24.7514801, 15.1584463, -34.9393196, 36.7657623
5: -14.7459335, 12.7260952, -18.6385632, 15.9667454, -30.7126751, 31.3646488
6: -15.7092972, 13.4410744, -19.8284340, 16.9537334, -32.6630325, 33.2695084
7: -17.7343292, 12.9453697, -22.2539272, 16.4043884, -34.1387138, 35.1992874
8: -20.4095192, 11.6607761, -25.7103214, 14.7018328, -35.1113510, 37.3710976
9: -14.2099781, 16.7049637, -17.9404049, 20.8574982, -35.0674744, 34.6453629

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1764142, upper bound: 27.1769159
time: 19.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1731947, upper bound: 27.1730404
time: 3.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.9134464, 13.5886955, -17.1321144, 13.0159340, -30.9293747, 30.7208099
1: -14.3370094, 12.1032810, -13.7113590, 11.6137381, -25.9507484, 25.8146400
2: -23.7386360, 7.8292828, -22.6482506, 7.6322775, -31.3709106, 30.4775333
3: -21.2624149, 9.6241035, -20.3188705, 9.2575092, -30.5199242, 29.9429741
4: -21.4619350, 13.0206089, -20.5076561, 12.4933710, -33.9553070, 33.5282669
5: -16.0196419, 13.8064632, -15.3415337, 13.2145271, -29.2341690, 29.1479969
6: -17.0694427, 14.5896454, -16.3475609, 13.9794044, -31.0488472, 30.9372025
7: -19.2575645, 14.0489292, -18.4233894, 13.4917746, -32.7493401, 32.4723129
8: -22.1685791, 12.6399736, -21.2236252, 12.1248455, -34.2934265, 33.8635979
9: -15.4282131, 18.1252384, -14.7862730, 17.3059444, -32.7341576, 32.9115105

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
time: 6.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
time: 4.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.6179695, 13.3672857, -21.1706066, 16.0676804, -33.6856461, 34.5378914
1: -14.0964661, 11.9054756, -17.0333099, 14.3833313, -28.4797974, 28.9387856
2: -23.3529835, 7.6948690, -27.7715530, 9.8016605, -33.1546440, 35.4664230
3: -20.9087181, 9.4687243, -25.2008877, 11.5280018, -32.4367218, 34.6696129
4: -21.1118050, 12.8082027, -25.2487450, 15.4905539, -36.6023560, 38.0569458
5: -15.7535038, 13.5823088, -19.0514755, 16.3041363, -32.0576363, 32.6337852
6: -16.7866821, 14.3503103, -20.2749901, 17.3030357, -34.0897102, 34.6252975
7: -18.9412766, 13.8186646, -22.7232590, 16.7723427, -35.7136192, 36.5419197
8: -21.8027096, 12.4340677, -26.2531052, 15.0228310, -36.8255386, 38.6871719
9: -15.1744480, 17.8298225, -18.3218479, 21.2848129, -36.4592590, 36.1516685

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1765486, upper bound: 27.1771004
time: 9.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1734917, upper bound: 27.1734233
time: 4.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.0928726, 12.9786110, -18.1451569, 13.7707014, -30.8635750, 31.1237679
1: -13.6705446, 11.5607567, -14.5273552, 12.2838631, -25.9544067, 26.0881119
2: -22.6523476, 7.5002651, -23.9761963, 8.0796852, -30.7320328, 31.4764576
3: -20.2838249, 9.2068090, -21.5348721, 9.7907639, -30.0745888, 30.7416801
4: -20.4843769, 12.4473114, -21.7123833, 13.2202616, -33.7046280, 34.1596909
5: -15.2870407, 13.1795158, -16.2447948, 13.9831371, -29.2701778, 29.4243107
6: -16.2841644, 13.9266672, -17.3167419, 14.7918262, -31.0759907, 31.2434044
7: -18.3731785, 13.4214592, -19.5018654, 14.2763662, -32.6495438, 32.9233246
8: -21.1478519, 12.0830069, -22.4712067, 12.8316269, -33.9794769, 34.5542145
9: -14.7271433, 17.2985573, -15.6506901, 18.3211784, -33.0483208, 32.9492455

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
time: 19.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
time: 23.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.7961121, 12.7564793, -21.8703003, 16.6020660, -33.3981781, 34.6267776
1: -13.4287777, 11.3624372, -17.5989552, 14.8497028, -28.2784805, 28.9613914
2: -22.2652435, 7.3662486, -28.6904755, 10.1303310, -32.3955765, 36.0567245
3: -19.9281845, 9.0508976, -26.0523186, 11.9043312, -31.8325157, 35.1032181
4: -20.1327095, 12.2344952, -26.0845413, 15.9975357, -36.1302414, 38.3190384
5: -15.0198059, 12.9544792, -19.6794357, 16.8346786, -31.8544846, 32.6339111
6: -16.0005703, 13.6864929, -20.9479122, 17.8660812, -33.8666496, 34.6344070
7: -18.0552540, 13.1901112, -23.4654331, 17.3205185, -35.3757706, 36.6555405
8: -20.7808342, 11.8769779, -27.1092682, 15.5209465, -36.3017807, 38.9862442
9: -14.4723549, 17.0018311, -18.9201736, 21.9886475, -36.4610023, 35.9220047

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1761392, upper bound: 27.1768080
time: 4.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1730766, upper bound: 27.1730009
time: 9.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18.2192554, 13.8199139, -18.5165920, 14.0555582, -32.2748146, 32.3365059
1: -14.5838032, 12.3127098, -14.8323431, 12.5382481, -27.1220512, 27.1450539
2: -24.1251945, 8.0010529, -24.4572105, 8.2749252, -32.4001160, 32.4582634
3: -21.6262245, 9.7959423, -21.9788895, 9.9940033, -31.6202278, 31.7748260
4: -21.8146820, 13.2459440, -22.1519356, 13.4925175, -35.3071976, 35.3978729
5: -16.2973003, 14.0371027, -16.5840836, 14.2682524, -30.5655518, 30.6211853
6: -17.3689480, 14.8372078, -17.6821938, 15.0975380, -32.4664841, 32.5194016
7: -19.5807228, 14.2970600, -19.8986435, 14.5749836, -34.1557045, 34.1957016
8: -22.5482597, 12.8634129, -22.9400272, 13.1008005, -35.6490593, 35.8034363
9: -15.6945677, 18.4227123, -15.9758968, 18.6910133, -34.3855820, 34.3986092

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
time: 5.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
time: 6.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.9213676, 13.5966873, -22.3314304, 16.9991894, -34.9205513, 35.9281158
1: -14.3414116, 12.1132145, -17.9944229, 15.1872368, -29.5286484, 30.1076374
2: -23.7364941, 7.8641362, -29.2784004, 10.4119701, -34.1484604, 37.1425362
3: -21.2700176, 9.6386070, -26.6234856, 12.1715689, -33.4415855, 36.2620926
4: -21.4623222, 13.0314684, -26.6571465, 16.3466854, -37.8090057, 39.6886101
5: -16.0287666, 13.8112116, -20.1347160, 17.2180138, -33.2467804, 33.9459267
6: -17.0835075, 14.5962162, -21.4078197, 18.2673264, -35.3508339, 36.0040359
7: -19.2619362, 14.0647793, -23.9656715, 17.7204056, -36.9823418, 38.0304451
8: -22.1791363, 12.6551447, -27.6938744, 15.8810539, -38.0601883, 40.3490181
9: -15.4386444, 18.1254406, -19.3482780, 22.4597321, -37.8983765, 37.4737053

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1763100, upper bound: 27.1770084
time: 7.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1733822, upper bound: 27.1733822
time: 4.11 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.54 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1725830, upper bound: 27.1728371
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1727857, upper bound: 27.1729945
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1746570, upper bound: 27.1750444
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909453
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1912932, upper bound: 27.1909477
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1722802, upper bound: 27.1726510
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1755090, upper bound: 27.1763456
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1725884, upper bound: 27.1728598
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1764142, upper bound: 27.1769159
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1731947, upper bound: 27.1730404
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1765486, upper bound: 27.1771004
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1734917, upper bound: 27.1734233
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1761392, upper bound: 27.1768080
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1730766, upper bound: 27.1730009
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1763100, upper bound: 27.1770084
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1733822, upper bound: 27.1733822

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.9401255, 12.0849628, -15.6298113, 11.8864088, -27.8265305, 27.7147732
1: -12.7191677, 10.7658005, -12.4897137, 10.5909882, -23.3101482, 23.2555084
2: -21.2377720, 6.7723961, -20.7523842, 6.8433084, -28.0810795, 27.5247803
3: -18.8964500, 8.5408030, -18.5304756, 8.4357395, -27.3321896, 27.0712776
4: -19.1518860, 11.5863152, -18.7578888, 11.4063644, -30.5582504, 30.3442020
5: -14.2089872, 12.3064327, -13.9771748, 12.0732813, -26.2822647, 26.2836075
6: -15.1642818, 12.9714594, -14.8856564, 12.7476292, -27.9119110, 27.8571110
7: -17.1360779, 12.4422522, -16.8171005, 12.2798710, -29.4159412, 29.2593498
8: -19.7277718, 11.2486038, -19.3510685, 11.0646057, -30.7923775, 30.5996723
9: -13.7084417, 16.1794186, -13.4754925, 15.8381977, -29.5466385, 29.6549110

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
time: 4.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
time: 4.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.2616520, 12.3257961, -16.7633343, 12.7316513, -28.9932976, 29.0891304
1: -12.9795742, 10.9794197, -13.4110098, 11.3436203, -24.3231926, 24.3904305
2: -21.6578560, 6.9174185, -22.2385750, 7.3291588, -28.9870148, 29.1559849
3: -19.2800903, 8.7088470, -19.8834877, 9.0235682, -28.3036575, 28.5923347
4: -19.5327034, 11.8155136, -20.0989819, 12.2082129, -31.7409172, 31.9144955
5: -14.4958162, 12.5507994, -14.9917889, 12.9358768, -27.4316902, 27.5425835
6: -15.4705286, 13.2300529, -15.9685974, 13.6642838, -29.1348114, 29.1986504
7: -17.4812393, 12.6928759, -18.0321121, 13.1566267, -30.6378670, 30.7249870
8: -20.1242046, 11.4713440, -20.7552090, 11.8434219, -31.9676266, 32.2265549
9: -13.9843311, 16.4995556, -14.4455919, 16.9747581, -30.9590855, 30.9451408

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
time: 10.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
time: 4.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18.9619484, 14.3820448, -15.3295527, 11.6617451, -30.6236916, 29.7115974
1: -15.1884060, 12.8066845, -12.2444410, 10.3909225, -25.5793285, 25.0511208
2: -25.1084862, 8.2703114, -20.3610001, 6.7097507, -31.8182373, 28.6313114
3: -22.5589066, 10.1774216, -18.1707001, 8.2786274, -30.8375340, 28.3481216
4: -22.7457809, 13.7782326, -18.4011192, 11.1912985, -33.9370804, 32.1793518
5: -16.9834347, 14.6120481, -13.7066708, 11.8452501, -28.8286743, 28.3187141
6: -18.0613613, 15.4357738, -14.5992002, 12.5039988, -30.5653534, 30.0349731
7: -20.3871632, 14.8479233, -16.4953308, 12.0465832, -32.4337463, 31.3432541
8: -23.4876900, 13.3483925, -18.9794693, 10.8569651, -34.3446541, 32.3278618
9: -16.3332138, 19.1900101, -13.2177391, 15.5371761, -31.8703804, 32.4077454

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
time: 6.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
time: 4.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18.3965988, 13.9618187, -16.4579887, 12.5029869, -30.8995781, 30.4198036
1: -14.7351627, 12.4361115, -13.1619911, 11.1399097, -25.8750706, 25.5981026
2: -24.3695984, 8.0422373, -21.8390884, 7.1928687, -31.5624657, 29.8813248
3: -21.8880806, 9.8874016, -19.5180187, 8.8640156, -30.7520943, 29.4054184
4: -22.0773525, 13.3835039, -19.7367096, 11.9895000, -34.0668526, 33.1202126
5: -16.4817219, 14.1808910, -14.7168732, 12.7039700, -29.1856918, 28.8977623
6: -17.5228195, 14.9826221, -15.6771812, 13.4170904, -30.9399033, 30.6598034
7: -19.7800751, 14.4156094, -17.7051010, 12.9191923, -32.6992645, 32.1207123
8: -22.7864819, 12.9652233, -20.3778305, 11.6318026, -34.4182854, 33.3430519
9: -15.8518791, 18.6240158, -14.1840363, 16.6682777, -32.5201569, 32.8080482

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1741871
time: 6.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1712147
time: 6.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -19.1809959, 14.5482531, -16.4579887, 12.5029869, -31.6839790, 31.0062408
1: -15.3690996, 12.9543724, -13.1619911, 11.1399097, -26.5090084, 26.1163597
2: -25.4019623, 8.3736982, -21.8390884, 7.1928687, -32.5948296, 30.2127800
3: -22.8216972, 10.2936506, -19.5180187, 8.8640156, -31.6857128, 29.8116684
4: -23.0056992, 13.9364824, -19.7367096, 11.9895000, -34.9951973, 33.6731911
5: -17.1814022, 14.7793951, -14.7168732, 12.7039700, -29.8853683, 29.4962654
6: -18.2712402, 15.6127682, -15.6771812, 13.4170904, -31.6883316, 31.2899494
7: -20.6236115, 15.0210228, -17.7051010, 12.9191923, -33.5428009, 32.7261238
8: -23.7586594, 13.5028687, -20.3778305, 11.6318026, -35.3904610, 33.8806992
9: -16.5217628, 19.4101315, -14.1840363, 16.6682777, -33.1900406, 33.5941658

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1743783
time: 5.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1714193
time: 23.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.2089806, 12.2873306, -16.9858341, 12.8967171, -29.1056976, 29.2731647
1: -12.9369707, 10.9443560, -13.5863924, 11.4882698, -24.4252357, 24.5307484
2: -21.5882263, 6.9008579, -22.5281563, 7.4377155, -29.0259399, 29.4290142
3: -19.2180462, 8.6837187, -20.1509781, 9.1429100, -28.3609543, 28.8346958
4: -19.4676609, 11.7806416, -20.3699417, 12.3765373, -31.8441982, 32.1505814
5: -14.4504128, 12.5096693, -15.1887407, 13.1009407, -27.5513535, 27.6984100
6: -15.4218426, 13.1868601, -16.1784515, 13.8394594, -29.2612991, 29.3653107
7: -17.4240055, 12.6549654, -18.2656956, 13.3297415, -30.7537384, 30.9206619
8: -20.0570927, 11.4375782, -21.0223770, 12.0031166, -32.0602036, 32.4599533
9: -13.9389820, 16.4470501, -14.6338034, 17.1963673, -31.1353493, 31.0808525

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
time: 7.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
time: 7.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.66 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1756639, upper bound: 27.1765012
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1758434, upper bound: 27.1766583
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1715719, upper bound: 27.1711670
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1741871
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1712147
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1752709, upper bound: 27.1743783
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1715664, upper bound: 27.1714193
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.66
Output dim: 2, lower bound: -27.1752181, upper bound: 27.1761430
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1755090, upper bound: 27.1763456
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1912625, upper bound: 27.1909302
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1912740, upper bound: 27.1909400
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1927327, upper bound: 27.1927070
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1764142, upper bound: 27.1769159
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1928051, upper bound: 27.1927796
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1765486, upper bound: 27.1771004
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1734917, upper bound: 27.1734233
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1927096, upper bound: 27.1927048
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1761392, upper bound: 27.1768080
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1927763, upper bound: 27.1927763
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1763100, upper bound: 27.1770084
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.66
Output dim: 2, lower bound: -27.1733822, upper bound: 27.1733822
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=36.862388610839844
rel_dist={2: [-27.20050538908776, 27.200505386462353]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1968311, upper bound: 27.1966598
time: 8.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987067, upper bound: 27.1987067
time: 7.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 2, lower bound: -27.1968311, upper bound: 27.1966598
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.84
Output dim: 2, lower bound: -27.1987067, upper bound: 27.1987067

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -17.7370834, 13.4382677, -18.8204269, 14.2666397, -32.0037231, 32.2586937
1: -14.1783390, 11.9624004, -15.0658188, 12.7009811, -26.8793182, 27.0282192
2: -23.5818024, 7.6063967, -24.9588394, 8.1844101, -31.7662125, 32.5652351
3: -21.0418549, 9.4901848, -22.3381920, 10.0795774, -31.1214333, 31.8283768
4: -21.2769165, 12.8763294, -22.5505219, 13.6658554, -34.9427719, 35.4268494
5: -15.8263702, 13.6713247, -16.8220825, 14.4953852, -30.3217545, 30.4934063
6: -16.8846321, 14.4183712, -17.9437828, 15.3111925, -32.1958199, 32.3621521
7: -19.0637054, 13.8504381, -20.2253342, 14.7256546, -33.7893600, 34.0757675
8: -21.9471779, 12.5008354, -23.3024693, 13.2719631, -35.2191391, 35.8032990
9: -15.2502155, 17.9744301, -16.1978798, 19.0492210, -34.2994385, 34.1723099

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1968310, upper bound: 27.1966566
time: 16.14 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1968310, upper bound: 27.1966598
time: 8.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -19.2813759, 14.6328573, -19.5879822, 14.8760109, -34.1573868, 34.2208405
1: -15.4504633, 13.0406036, -15.7055073, 13.2615089, -28.7119713, 28.7461109
2: -25.4806252, 8.5678139, -25.8488884, 8.7850285, -34.2656479, 34.4166985
3: -22.9028149, 10.3804493, -23.2756615, 10.5697727, -33.4725838, 33.6561050
4: -23.0672989, 14.0238628, -23.4238701, 14.2618408, -37.3291397, 37.4477310
5: -17.2664795, 14.8502493, -17.5568924, 15.0878859, -32.3543663, 32.4071426
6: -18.4147415, 15.7105303, -18.7215919, 15.9716549, -34.3863907, 34.4321213
7: -20.7114697, 15.1501713, -21.0358124, 15.4171734, -36.1286392, 36.1859779
8: -23.8846569, 13.6342249, -24.2695675, 13.8700180, -37.7546692, 37.9037933
9: -16.6258526, 19.4761505, -16.9053860, 19.7695770, -36.3954315, 36.3815384

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1986938, upper bound: 27.1986928
time: 36.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987067, upper bound: 27.1987067
time: 6.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 45.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 45.16
Output dim: 2, lower bound: -27.1968310, upper bound: 27.1966566
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 45.16
Output dim: 2, lower bound: -27.1968310, upper bound: 27.1966598
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 45.16
Output dim: 2, lower bound: -27.1986938, upper bound: 27.1986928
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 45.16
Output dim: 2, lower bound: -27.1987067, upper bound: 27.1987067

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -17.0300217, 12.8998518, -17.0075703, 12.9007215, -29.9307404, 29.9074211
1: -13.6032963, 11.4895267, -13.5937948, 11.4884319, -25.0917282, 25.0833206
2: -22.6717377, 7.2524714, -22.6071167, 7.3227024, -29.9944344, 29.8595867
3: -20.1982269, 9.1067486, -20.1718216, 9.1207466, -29.3189735, 29.2785664
4: -20.4491215, 12.3621683, -20.4062843, 12.3578644, -32.8069839, 32.7684517
5: -15.1819038, 13.1357822, -15.1838751, 13.1225014, -28.3043976, 28.3196564
6: -16.2011948, 13.8493547, -16.1898708, 13.8407192, -30.0419140, 30.0392265
7: -18.3085136, 13.2861633, -18.2902222, 13.3020802, -31.6105881, 31.5763836
8: -21.0757790, 12.0004158, -21.0545807, 11.9943886, -33.0701675, 33.0549927
9: -14.6422348, 17.2696266, -14.6380386, 17.2341824, -31.8764153, 31.9076653

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1913441, upper bound: 27.1914419
time: 7.15 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1910300, upper bound: 27.1908487
time: 7.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.4212093, 13.1967487, -18.3395271, 13.8963451, -31.3175545, 31.5362740
1: -13.9217873, 11.7510166, -14.6722698, 12.3710394, -26.2928257, 26.4232864
2: -23.1817722, 7.4423008, -24.3540726, 7.9116893, -31.0934620, 31.7963734
3: -20.6662788, 9.3164101, -21.7636509, 9.8136463, -30.4799252, 31.0800533
4: -20.9109383, 12.6458397, -21.9926033, 13.3109045, -34.2218437, 34.6384354
5: -15.5369959, 13.4326963, -16.3785553, 14.1321411, -29.6691360, 29.8112526
6: -16.5764370, 14.1643219, -17.4646816, 14.9146271, -31.4910641, 31.6290016
7: -18.7286606, 13.5968723, -19.7157555, 14.3346901, -33.0633507, 33.3126221
8: -21.5573120, 12.2750578, -22.7003517, 12.9182348, -34.4755478, 34.9754105
9: -14.9783268, 17.6606102, -15.7767324, 18.5718002, -33.5501213, 33.4373436

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1913139, upper bound: 27.1914243
time: 7.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1910133, upper bound: 27.1908462
time: 6.17 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -18.4092293, 13.9606142, -17.6260433, 13.3833218, -31.7925453, 31.5866585
1: -14.7376156, 12.4351625, -14.1104469, 11.9292784, -26.6668930, 26.5456085
2: -24.3860645, 8.0545311, -23.3329601, 7.7802911, -32.1663551, 31.3874912
3: -21.8450699, 9.8789539, -20.9077625, 9.4903154, -31.3353844, 30.7867165
4: -22.0535240, 13.3750029, -21.1091499, 12.8297710, -34.8832932, 34.4841537
5: -16.4662476, 14.1849327, -15.7773285, 13.5904560, -30.0567017, 29.9622612
6: -17.5521355, 14.9922314, -16.8122921, 14.3703461, -31.9224777, 31.8045235
7: -19.7915955, 14.4318151, -18.9557209, 13.8498878, -33.6414833, 33.3875313
8: -22.7957897, 12.9844589, -21.8334274, 12.4539537, -35.2497444, 34.8178787
9: -15.8554430, 18.6202946, -15.1982784, 17.8203411, -33.6757812, 33.8185654

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1981677, upper bound: 27.1981356
time: 5.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1986322, upper bound: 27.1986276
time: 4.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -18.8701935, 14.3140259, -18.9989796, 14.4143286, -33.2845230, 33.3130035
1: -15.1137409, 12.7531309, -15.2224865, 12.8478298, -27.9615669, 27.9756165
2: -24.9728966, 8.3161516, -25.1220455, 8.4204149, -33.3933029, 33.4381943
3: -22.4022179, 10.1395760, -22.5583076, 10.2220278, -32.6242447, 32.6978836
4: -22.5916615, 13.7162962, -22.7394714, 13.8205061, -36.4121666, 36.4557686
5: -16.8874626, 14.5357914, -17.0100632, 14.6351376, -31.5225945, 31.5458546
6: -18.0054131, 15.3704300, -18.1356068, 15.4813871, -33.4868011, 33.5060349
7: -20.2808628, 14.8077164, -20.4187965, 14.9235973, -35.2044525, 35.2265091
8: -23.3711300, 13.3242683, -23.5360718, 13.4230900, -36.7942200, 36.8603401
9: -16.2603760, 19.0740070, -16.3793411, 19.1933231, -35.4536896, 35.4533386

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1982038, upper bound: 27.1981734
time: 18.95 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1986820, upper bound: 27.1986820
time: 6.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1913441, upper bound: 27.1914419
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1910300, upper bound: 27.1908487
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1913139, upper bound: 27.1914243
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1910133, upper bound: 27.1908462
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1981677, upper bound: 27.1981356
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1986322, upper bound: 27.1986276
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1982038, upper bound: 27.1981734
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.53
Output dim: 2, lower bound: -27.1986820, upper bound: 27.1986820

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.4490433, 12.4647636, -16.5700111, 12.5723648, -29.0214062, 29.0347652
1: -13.1313963, 11.1041374, -13.2375011, 11.1976233, -24.3290157, 24.3416386
2: -21.9096584, 6.9940810, -22.0343361, 7.1265440, -29.0362015, 29.0284176
3: -19.5027161, 8.8030224, -19.6478081, 8.8918791, -28.3945961, 28.4508305
4: -19.7591419, 11.9473572, -19.8868065, 12.0448570, -31.8039970, 31.8341637
5: -14.6603508, 12.6947174, -14.7894459, 12.7906523, -27.4510040, 27.4841633
6: -15.6471729, 13.3820581, -15.7721043, 13.4874315, -29.1346035, 29.1541595
7: -17.6856422, 12.8352365, -17.8212566, 12.9612198, -30.6468620, 30.6564922
8: -20.3588371, 11.5985088, -20.5143909, 11.6912861, -32.0501251, 32.1128960
9: -14.1454935, 16.6874924, -14.2627392, 16.7959938, -30.9414825, 30.9502316

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912053, upper bound: 27.1913314
time: 10.04 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1912145, upper bound: 27.1913419
time: 7.57 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -19.4691391, 14.7643032, -16.2194214, 12.3096952, -31.7788353, 30.9837227
1: -15.6017857, 13.1466599, -12.9517660, 10.9644947, -26.5662804, 26.0984230
2: -25.7783909, 8.4959908, -21.5759144, 6.9733539, -32.7517433, 30.0719051
3: -23.1627045, 10.4421177, -19.2274990, 8.7095623, -31.8722668, 29.6696167
4: -23.3536224, 14.1394510, -19.4697819, 11.7947922, -35.1484146, 33.6092300
5: -17.4382095, 15.0015144, -14.4737606, 12.5240841, -29.9622917, 29.4752731
6: -18.5440865, 15.8469982, -15.4377604, 13.2038975, -31.7479839, 31.2847557
7: -20.9369717, 15.2427454, -17.4453030, 12.6892977, -33.6262703, 32.6880455
8: -24.1210022, 13.6971779, -20.0805130, 11.4496727, -35.5706673, 33.7776871
9: -16.7706032, 19.6978970, -13.9620447, 16.4440651, -33.2146683, 33.6599388

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908919, upper bound: 27.1907371
time: 6.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908969, upper bound: 27.1907423
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.8518486, 12.7701683, -17.9046879, 13.5696745, -30.4215240, 30.6748543
1: -13.4593372, 11.3731537, -14.3179493, 12.0807867, -25.5401230, 25.6911030
2: -22.4360218, 7.1887617, -23.7849007, 7.7139087, -30.1499310, 30.9736557
3: -19.9843273, 9.0183849, -21.2433357, 9.5858746, -29.5702019, 30.2617207
4: -20.2347107, 12.2393970, -21.4771347, 12.9996309, -33.2343407, 33.7165298
5: -15.0251789, 13.0004244, -15.9861813, 13.8025455, -28.8277245, 28.9866066
6: -16.0336189, 13.7061968, -17.0476475, 14.5631990, -30.5968170, 30.7538452
7: -18.1185284, 13.1550941, -19.2503891, 13.9955072, -32.1140327, 32.4054832
8: -20.8548965, 11.8815155, -22.1618652, 12.6150246, -33.4699211, 34.0433807
9: -14.4918346, 17.0897713, -15.4035225, 18.1359882, -32.6278191, 32.4932938

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1911824, upper bound: 27.1913174
time: 7.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1911855, upper bound: 27.1913262
time: 4.87 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -19.8919334, 15.0855160, -17.5548458, 13.3066130, -33.1985435, 32.6403580
1: -15.9459820, 13.4293594, -14.0328903, 11.8468018, -27.7927837, 27.4622498
2: -26.3282738, 8.7007351, -23.3252678, 7.5564008, -33.8846703, 32.0260010
3: -23.6677628, 10.6698427, -20.8245544, 9.4034100, -33.0711746, 31.4943924
4: -23.8512955, 14.4456406, -21.0609436, 12.7490234, -36.6003151, 35.5065842
5: -17.8221512, 15.3228083, -15.6704311, 13.5366554, -31.3588066, 30.9932404
6: -18.9497414, 16.1874313, -16.7119446, 14.2797737, -33.2295151, 32.8993759
7: -21.3906403, 15.5789890, -18.8751068, 13.7229614, -35.1136017, 34.4540939
8: -24.6415405, 13.9938755, -21.7272587, 12.3714218, -37.0129623, 35.7211342
9: -17.1339149, 20.1197033, -15.1029329, 17.7845440, -34.9184494, 35.2226334

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908789, upper bound: 27.1907354
time: 33.11 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908815, upper bound: 27.1907402
time: 18.34 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.5869331, 12.5949516, -16.5437870, 12.5695229, -29.1564560, 29.1387367
1: -13.2580624, 11.2142162, -13.2318840, 11.1967106, -24.4547729, 24.4460983
2: -22.0155392, 7.2184501, -21.9411736, 7.2420230, -29.2575626, 29.1596184
3: -19.6751041, 8.9215164, -19.6227646, 8.9107199, -28.5858231, 28.5442810
4: -19.8992004, 12.0771723, -19.8341331, 12.0485973, -31.9477921, 31.9113045
5: -14.8275003, 12.7971935, -14.7995949, 12.7667656, -27.5942650, 27.5967884
6: -15.7957230, 13.5138760, -15.7611713, 13.4896898, -29.2854118, 29.2750473
7: -17.8380451, 13.0064259, -17.7975883, 12.9923410, -30.8303871, 30.8040142
8: -20.5277882, 11.7168465, -20.4825497, 11.6908417, -32.2186279, 32.1993904
9: -14.2862511, 16.8031788, -14.2594156, 16.7524948, -31.0387459, 31.0625954

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1932356, upper bound: 27.1929693
time: 7.92 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1908956, upper bound: 27.1926543
time: 7.80 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.7173615, 13.4391994, -17.0873947, 12.9767933, -30.6941547, 30.5265903
1: -14.1775112, 11.9661875, -13.6744871, 11.5631046, -25.7406158, 25.6406746
2: -23.4981194, 7.7060633, -22.6433792, 7.5046859, -31.0028019, 30.3494415
3: -21.0255470, 9.5084572, -20.2692680, 9.1997757, -30.2253208, 29.7777252
4: -21.2376614, 12.8750401, -20.4750862, 12.4392614, -33.6769218, 33.3501205
5: -15.8402157, 13.6580791, -15.2894669, 13.1804867, -29.0207024, 28.9475460
6: -16.8774529, 14.4280005, -16.2840614, 13.9308939, -30.8083458, 30.7120590
7: -19.0511208, 13.8831263, -18.3805237, 13.4222145, -32.4733353, 32.2636490
8: -21.9284134, 12.4935493, -21.1561127, 12.0694714, -33.9978867, 33.6496582
9: -15.2544889, 17.9367371, -14.7290993, 17.2895622, -32.5440521, 32.6658249

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1933072, upper bound: 27.1930537
time: 7.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927475, upper bound: 27.1927348
time: 5.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17.0417175, 12.9404259, -17.9246960, 13.5976753, -30.6393890, 30.8651218
1: -13.6304493, 11.5232353, -14.3459301, 12.1146059, -25.7450562, 25.8691654
2: -22.6012573, 7.4543099, -23.7409229, 7.8626533, -30.4639034, 31.1952305
3: -20.2221985, 9.1708927, -21.2775574, 9.6385880, -29.8607864, 30.4484501
4: -20.4320717, 12.4079685, -21.4715462, 13.0372343, -33.4693069, 33.8795052
5: -15.2407494, 13.1421795, -16.0337257, 13.8132896, -29.0540314, 29.1759052
6: -16.2323685, 13.8848534, -17.0805206, 14.6006079, -30.8329697, 30.9653740
7: -18.3229561, 13.3738060, -19.2712173, 14.0663328, -32.3892899, 32.6450195
8: -21.0876617, 12.0423107, -22.1829300, 12.6508217, -33.7384834, 34.2252426
9: -14.6804972, 17.2547989, -15.4418488, 18.1288471, -32.8093452, 32.6966476

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1932374, upper bound: 27.1929771
time: 6.37 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1926556, upper bound: 27.1926537
time: 4.61 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18.1653786, 13.7791204, -18.4660378, 14.0071144, -32.1724930, 32.2451591
1: -14.5423670, 12.2724380, -14.7880688, 12.4833870, -27.0257530, 27.0605068
2: -24.0711937, 7.9484921, -24.4388142, 8.1393070, -32.2105026, 32.3873062
3: -21.5632553, 9.7573128, -21.9201946, 9.9312801, -31.4945297, 31.6774960
4: -21.7609386, 13.2027836, -22.1088314, 13.4310913, -35.1920242, 35.3116150
5: -16.2477684, 13.9980631, -16.5246506, 14.2266645, -30.4744301, 30.5227108
6: -17.3112984, 14.7934380, -17.6092472, 15.0426731, -32.3539658, 32.4026871
7: -19.5287075, 14.2470427, -19.8498554, 14.4974642, -34.0261726, 34.0968933
8: -22.4815826, 12.8170910, -22.8617172, 13.0374889, -35.5190735, 35.6788025
9: -15.6450348, 18.3770504, -15.9126472, 18.6636162, -34.3086510, 34.2896957

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1933080, upper bound: 27.1930589
time: 6.94 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1927337, upper bound: 27.1927337
time: 16.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1912053, upper bound: 27.1913314
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1912145, upper bound: 27.1913419
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1908919, upper bound: 27.1907371
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1908969, upper bound: 27.1907423
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1911824, upper bound: 27.1913174
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1911855, upper bound: 27.1913262
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1908789, upper bound: 27.1907354
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1908815, upper bound: 27.1907402
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1932356, upper bound: 27.1929693
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1908956, upper bound: 27.1926543
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1933072, upper bound: 27.1930537
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1927475, upper bound: 27.1927348
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1932374, upper bound: 27.1929771
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1926556, upper bound: 27.1926537
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1933080, upper bound: 27.1930589
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 2, lower bound: -27.1927337, upper bound: 27.1927337

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.4978495, 11.7539749, -14.7837563, 11.2381296, -26.7359734, 26.5377312
1: -12.3630810, 10.4738121, -11.7927999, 10.0156784, -22.3787537, 22.2666111
2: -20.6692200, 6.5677328, -19.6987495, 6.3572645, -27.0264854, 26.2664795
3: -18.3671265, 8.3065529, -17.5180721, 7.9631410, -26.3302650, 25.8246250
4: -18.6319389, 11.2702360, -17.7714329, 10.7824268, -29.4143658, 29.0416679
5: -13.8123035, 11.9725399, -13.1976128, 11.4298687, -25.2421722, 25.1701527
6: -14.7418842, 12.6171989, -14.0703259, 12.0503101, -26.7921925, 26.6875248
7: -16.6645069, 12.0956459, -15.9033318, 11.5795364, -28.2440434, 27.9989777
8: -19.1854858, 10.9417582, -18.3025951, 10.4664030, -29.6518879, 29.2443542
9: -13.3298197, 15.7405624, -12.7338696, 15.0108414, -28.3406601, 28.4744320

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1725346, upper bound: 27.1728878
time: 7.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1714177, upper bound: 27.1715334
time: 8.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.9934158, 12.1242466, -15.9194584, 12.0819502, -28.0753613, 28.0437050
1: -12.7648373, 10.8029804, -12.7105732, 10.7651443, -23.5299797, 23.5135536
2: -21.3192024, 6.7883625, -21.1908989, 6.8293872, -28.1485901, 27.9792614
3: -18.9612560, 8.5650921, -18.8767452, 8.5494967, -27.5107536, 27.4418373
4: -19.2202854, 11.6236382, -19.1154671, 11.5811672, -30.8014488, 30.7391052
5: -14.2550745, 12.3486977, -14.2054796, 12.2940006, -26.5490723, 26.5541725
6: -15.2139874, 13.0159683, -15.1514797, 12.9620466, -28.1760330, 28.1674480
7: -17.1963539, 12.4809589, -17.1209297, 12.4516573, -29.6480103, 29.6018791
8: -19.7969818, 11.2848482, -19.7073288, 11.2439842, -31.0409641, 30.9921761
9: -13.7547932, 16.2351112, -13.7030191, 16.1479282, -29.9027214, 29.9381294

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1727966, upper bound: 27.1730915
time: 7.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716707, upper bound: 27.1717293
time: 5.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18.5101967, 14.0401278, -14.4306736, 10.9758673, -29.4860592, 28.4708023
1: -14.8223438, 12.5057163, -11.5063124, 9.7818260, -24.6041679, 24.0120277
2: -24.5296249, 8.0586815, -19.2377338, 6.2062511, -30.7358761, 27.2964153
3: -22.0211563, 9.9350977, -17.0937500, 7.7803302, -29.8014851, 27.0288467
4: -22.2133732, 13.4542351, -17.3518066, 10.5322790, -32.7456512, 30.8060417
5: -16.5743523, 14.2680025, -12.8826084, 11.1621151, -27.7364655, 27.1506119
6: -17.6279526, 15.0722132, -13.7345486, 11.7661438, -29.3940964, 28.8067627
7: -19.9042397, 14.4909916, -15.5254717, 11.3078556, -31.2120953, 30.0164642
8: -22.9305077, 13.0353508, -17.8669987, 10.2238379, -33.1543388, 30.9023476
9: -15.9451361, 18.7403603, -12.4321766, 14.6573467, -30.6024818, 31.1725311

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1721196, upper bound: 27.1723108
time: 15.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1710455, upper bound: 27.1708891
time: 6.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -19.0045700, 14.4136562, -15.5517120, 11.8078442, -30.8124142, 29.9653645
1: -15.2250500, 12.8366241, -12.4114761, 10.5208530, -25.7459030, 25.2480965
2: -25.1754932, 8.2837534, -20.7087517, 6.6702294, -31.8457222, 28.9925041
3: -22.6101036, 10.1963902, -18.4355831, 8.3585014, -30.9686050, 28.6319714
4: -22.8011684, 13.8076382, -18.6779404, 11.3198547, -34.1210251, 32.4855766
5: -17.0198517, 14.6463070, -13.8774128, 12.0141010, -29.0339432, 28.5237179
6: -18.1005859, 15.4715385, -14.8012705, 12.6657782, -30.7663651, 30.2728081
7: -20.4362278, 14.8786869, -16.7265339, 12.1676178, -32.6038437, 31.6052208
8: -23.5440865, 13.3772383, -19.2521896, 10.9908113, -34.5348969, 32.6294174
9: -16.3705883, 19.2344379, -13.3883066, 15.7794247, -32.1500130, 32.6227455

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723899, upper bound: 27.1725309
time: 8.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1713289, upper bound: 27.1711304
time: 6.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.9060116, 12.0601101, -16.1392708, 12.2448750, -28.1508846, 28.1993790
1: -12.6938629, 10.7448349, -12.8859339, 10.9072495, -23.6011124, 23.6307678
2: -21.2048206, 6.7591414, -21.4793606, 6.9359603, -28.1407814, 28.2385025
3: -18.8575439, 8.5223627, -19.1393623, 8.6637955, -27.5213394, 27.6617241
4: -19.1139183, 11.5654402, -19.3848953, 11.7472429, -30.8611584, 30.9503365
5: -14.1789551, 12.2807770, -14.4010277, 12.4572124, -26.6361656, 26.6818008
6: -15.1325493, 12.9447174, -15.3589764, 13.1369896, -28.2695389, 28.3036938
7: -17.1020985, 12.4170971, -17.3538170, 12.6230488, -29.7251472, 29.7709084
8: -19.6863251, 11.2281761, -19.9712219, 11.4004040, -31.0867290, 31.1993980
9: -13.6797304, 16.1479149, -13.8886147, 16.3692360, -30.0489616, 30.0365295

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1722828, upper bound: 27.1726954
time: 10.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1712875, upper bound: 27.1714452
time: 10.22 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.4100685, 12.4377441, -17.2698021, 13.0907078, -29.5007744, 29.7075462
1: -13.1023331, 11.0798302, -13.8029718, 11.6562309, -24.7585640, 24.8828011
2: -21.8637733, 6.9861465, -22.9641361, 7.4138536, -29.2776241, 29.9502811
3: -19.4594345, 8.7858229, -20.4914398, 9.2510433, -28.7104778, 29.2772636
4: -19.7104721, 11.9245052, -20.7245598, 12.5456152, -32.2560844, 32.6490631
5: -14.6288652, 12.6638231, -15.4135752, 13.3188200, -27.9476833, 28.0773983
6: -15.6126270, 13.3499756, -16.4378128, 14.0477314, -29.6603584, 29.7877884
7: -17.6430073, 12.8099966, -18.5678177, 13.4958115, -31.1388187, 31.3778152
8: -20.3080025, 11.5773525, -21.3721008, 12.1756811, -32.4836845, 32.9494476
9: -14.1121693, 16.6500263, -14.8555412, 17.5040283, -31.6161919, 31.5055676

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1725677, upper bound: 27.1729220
time: 6.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1715682, upper bound: 27.1716552
time: 7.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18.9378910, 14.3646641, -15.7881346, 11.9828377, -30.9207287, 30.1527977
1: -15.1704998, 12.7913494, -12.6003180, 10.6741257, -25.8446255, 25.3916664
2: -25.0863037, 8.2639580, -21.0189323, 6.7832494, -31.8695488, 29.2828884
3: -22.5323906, 10.1647472, -18.7178230, 8.4812527, -31.0136433, 28.8825684
4: -22.7176323, 13.7636757, -18.9669495, 11.4974041, -34.2150345, 32.7306252
5: -16.9625130, 14.5929461, -14.0870686, 12.1903648, -29.1528759, 28.6800137
6: -18.0382481, 15.4165478, -15.0246086, 12.8538857, -30.8921280, 30.4411545
7: -20.3627014, 14.8302183, -16.9770546, 12.3515863, -32.7142868, 31.8072624
8: -23.4570351, 13.3352385, -19.5370598, 11.1582346, -34.6152687, 32.8722954
9: -16.3124886, 19.1674747, -13.5881710, 16.0171852, -32.3296738, 32.7556458

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1719338, upper bound: 27.1722064
time: 6.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1709738, upper bound: 27.1708570
time: 86.08 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -19.4396133, 14.7440996, -16.9075432, 12.8184328, -32.2580414, 31.6516418
1: -15.5791645, 13.1273928, -13.5074987, 11.4148932, -26.9940567, 26.6348915
2: -25.7416229, 8.4935770, -22.4886971, 7.2531681, -32.9947891, 30.9822712
3: -23.1296539, 10.4303761, -20.0570621, 9.0621605, -32.1918144, 30.4874306
4: -23.3134842, 14.1224346, -20.2930107, 12.2862749, -35.5997581, 34.4154396
5: -17.4147758, 14.9769964, -15.0868015, 13.0431509, -30.4579258, 30.0637970
6: -18.5179272, 15.8217678, -16.0921612, 13.7544079, -32.2723351, 31.9139252
7: -20.9029198, 15.2241821, -18.1787033, 13.2138042, -34.1167221, 33.4028854
8: -24.0798321, 13.6822720, -20.9237862, 11.9250002, -36.0048256, 34.6060562
9: -16.7442551, 19.6686249, -14.5445614, 17.1403046, -33.8845520, 34.2131882

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 249

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1722194, upper bound: 27.1724368
time: 8.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1712803, upper bound: 27.1711066
time: 4.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.1447868, 12.2636375, -15.9083176, 12.0935535, -28.2383347, 28.1719513
1: -12.8972874, 10.9197502, -12.7139101, 10.7725964, -23.6698780, 23.6336594
2: -21.4372406, 7.0189824, -21.1100864, 6.9537296, -28.3909702, 28.1290684
3: -19.1450844, 8.6900835, -18.8614807, 8.5774307, -27.7225151, 27.5515633
4: -19.3743439, 11.7606831, -19.0807056, 11.5930271, -30.9673710, 30.8413868
5: -14.4287376, 12.4620495, -14.2271099, 12.2848186, -26.7135506, 26.6891556
6: -15.3734798, 13.1564426, -15.1545286, 12.9757023, -28.3491821, 28.3109703
7: -17.3642883, 12.6619997, -17.1172485, 12.4971237, -29.8614082, 29.7792473
8: -19.9817085, 11.4107037, -19.6982307, 11.2494354, -31.2311440, 31.1089344
9: -13.9071741, 16.3600941, -13.7147655, 16.1158104, -30.0229836, 30.0748596

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746481, upper bound: 27.1746231
time: 6.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1735451, upper bound: 27.1733231
time: 3.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.8084545, 12.0119448, -19.6636295, 14.9066105, -30.7150631, 31.6755753
1: -12.6226425, 10.6959248, -15.7802181, 13.3341007, -25.9567432, 26.4761410
2: -20.9984131, 6.8713312, -25.8557243, 8.8994160, -29.8978291, 32.7270546
3: -18.7419128, 8.5151024, -23.3887939, 10.6452637, -29.3871746, 31.9038963
4: -18.9740143, 11.5208187, -23.4994965, 14.3523502, -33.3263626, 35.0203171
5: -14.1256886, 12.2065182, -17.6414986, 15.1386080, -29.2642975, 29.8480167
6: -15.0523520, 12.8839684, -18.7614193, 16.0585823, -31.1109352, 31.6453876
7: -17.0037155, 12.4010086, -21.1260872, 15.5073814, -32.5110970, 33.5270920
8: -19.5653229, 11.1791077, -24.3643970, 13.8955078, -33.4608307, 35.5434990
9: -13.6186981, 16.0225544, -16.9889488, 19.8081512, -33.4268494, 33.0114975

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 249

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1722266, upper bound: 27.1743776
time: 8.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1729905, upper bound: 27.1729208
time: 11.11 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.2805328, 13.1119919, -16.4590588, 12.5070362, -29.7875690, 29.5710506
1: -13.8218899, 11.6751251, -13.1639938, 11.1439190, -24.9658089, 24.8391151
2: -22.9273205, 7.5082717, -21.8248234, 7.2179484, -30.1452694, 29.3330956
3: -20.5017815, 9.2796059, -19.5163765, 8.8694305, -29.3712120, 28.7959747
4: -20.7199821, 12.5624790, -19.7321701, 11.9889393, -32.7089233, 32.2946472
5: -15.4466496, 13.3269768, -14.7237692, 12.7053070, -28.1519566, 28.0507355
6: -16.4604073, 14.0753870, -15.6835718, 13.4241905, -29.8845978, 29.7589588
7: -18.5837326, 13.5426922, -17.7091808, 12.9322910, -31.5160141, 31.2518730
8: -21.3895664, 12.1904593, -20.3811493, 11.6331358, -33.0227013, 32.5716095
9: -14.8806143, 17.4990463, -14.1906633, 16.6624374, -31.5430527, 31.6897087

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1749670, upper bound: 27.1749852
time: 10.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1739509, upper bound: 27.1737458
time: 8.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.21 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1725346, upper bound: 27.1728878
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1714177, upper bound: 27.1715334
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1727966, upper bound: 27.1730915
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1716707, upper bound: 27.1717293
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1721196, upper bound: 27.1723108
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1710455, upper bound: 27.1708891
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1723899, upper bound: 27.1725309
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1713289, upper bound: 27.1711304
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1722828, upper bound: 27.1726954
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1712875, upper bound: 27.1714452
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1725677, upper bound: 27.1729220
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1715682, upper bound: 27.1716552
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1719338, upper bound: 27.1722064
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1709738, upper bound: 27.1708570
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1722194, upper bound: 27.1724368
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1712803, upper bound: 27.1711066
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1746481, upper bound: 27.1746231
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1735451, upper bound: 27.1733231
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1722266, upper bound: 27.1743776
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1729905, upper bound: 27.1729208
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1749670, upper bound: 27.1749852
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.21
Output dim: 2, lower bound: -27.1739509, upper bound: 27.1737458
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.21
Output dim: 2, lower bound: -27.1927475, upper bound: 27.1927348
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.21
Output dim: 2, lower bound: -27.1932374, upper bound: 27.1929771
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.21
Output dim: 2, lower bound: -27.1926556, upper bound: 27.1926537
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.21
Output dim: 2, lower bound: -27.1933080, upper bound: 27.1930589
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.21
Output dim: 2, lower bound: -27.1927337, upper bound: 27.1927337
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=36.862388610839844
rel_dist={2: [-27.200440708568877, 27.200440707169562]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1858.45 seconds
