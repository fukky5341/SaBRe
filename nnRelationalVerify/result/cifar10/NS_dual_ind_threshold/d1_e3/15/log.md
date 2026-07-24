## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 15)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0802089108


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0987847, -2.0332274, -3.0987847, -2.0332274, -0.3889209, 0.3889208)
1: (-3.8316810, -2.3243079, -3.8316810, -2.3243079, -0.5720347, 0.5720347)
2: (-1.0117517, -0.5120511, -1.0117517, -0.5120511, -0.2019678, 0.2019678)
3: (-1.8510038, -1.4167778, -1.8510038, -1.4167778, -0.1251124, 0.1251123)
4: (0.1425383, 0.3805816, 0.1425383, 0.3805816, -0.1757070, 0.1757070)
5: (-1.4793326, -1.0883714, -1.4793326, -1.0883714, -0.1334287, 0.1334287)
6: (-0.6454182, -0.0373911, -0.6454182, -0.0373911, -0.1606191, 0.1606191)
7: (0.1194437, 0.7281401, 0.1194437, 0.7281401, -0.4625651, 0.4625652)
8: (-4.5755615, -3.7448506, -4.5755615, -3.7448506, -0.4125719, 0.4125719)
9: (-4.7688985, -3.9082353, -4.7688985, -3.9082353, -0.3464887, 0.3464887)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.89 + 122.67 = 130.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0802878, upper bound: 0.0802884

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 280
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 266
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 3330
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2837
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3213
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 3570
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 3581
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3317
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 3332
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 280

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802843, upper bound: 0.0800672
time: 14.93 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802852, upper bound: 0.0800679
time: 131.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 146.31 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 146.31
Output dim: 7, lower bound: -0.0802843, upper bound: 0.0800672
NS_A2, status: Status.UNKNOWN, split count: 1, time: 146.31
Output dim: 7, lower bound: -0.0802852, upper bound: 0.0800679

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.0954695, -2.0346160, -3.0971823, -2.0334814, -0.3840055, 0.3851082
1: -3.8310020, -2.3242791, -3.8311334, -2.3243089, -0.5713208, 0.5716225
2: -1.0095840, -0.5155813, -1.0098310, -0.5126436, -0.1990374, 0.1963255
3: -1.8495977, -1.4181634, -1.8508894, -1.4180000, -0.1221517, 0.1234113
4: 0.1466032, 0.3784804, 0.1433750, 0.3803594, -0.1702869, 0.1733610
5: -1.4780431, -1.0899910, -1.4793262, -1.0895813, -0.1306112, 0.1316824
6: -0.6427684, -0.0394315, -0.6431172, -0.0373955, -0.1579452, 0.1561498
7: 0.1227255, 0.7231141, 0.1194898, 0.7237682, -0.4547812, 0.4574083
8: -4.5741749, -3.7448230, -4.5751214, -3.7448559, -0.4109547, 0.4117835
9: -4.7655053, -3.9117901, -4.7688866, -3.9113226, -0.3400841, 0.3429020

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 266
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 3500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 280
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3330
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2837
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 3213
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 3570
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3581
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3317
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 3332
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 334

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0800700, upper bound: 0.0800663
time: 237.09 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802829, upper bound: 0.0800663
time: 291.14 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.0981750, -2.0332332, -3.0982518, -2.0332325, -0.3859199, 0.3882753
1: -3.8313117, -2.3243084, -3.8313584, -2.3243084, -0.5718638, 0.5717244
2: -1.0117356, -0.5120607, -1.0117373, -0.5120594, -0.1977085, 0.2019207
3: -1.8510013, -1.4168491, -1.8510014, -1.4168425, -0.1250253, 0.1223995
4: 0.1425722, 0.3805770, 0.1425683, 0.3805777, -0.1740548, 0.1747673
5: -1.4793324, -1.0885115, -1.4793324, -1.0884957, -0.1334021, 0.1307007
6: -0.6453727, -0.0373921, -0.6453784, -0.0373920, -0.1563363, 0.1605671
7: 0.1194475, 0.7280911, 0.1194475, 0.7280967, -0.4625144, 0.4594253
8: -4.5752425, -3.7448523, -4.5752759, -3.7448521, -0.4121001, 0.4122396
9: -4.7688985, -3.9082563, -4.7688985, -3.9082537, -0.3464315, 0.3400115

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 266
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 3500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 280
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 3330
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2837
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 3213
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 3570
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3581
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3317
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 3332
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 334

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0800703, upper bound: 0.0802839
time: 12.26 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802835, upper bound: 0.0800669
time: 144.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 163.02 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 163.02
Output dim: 7, lower bound: -0.0800700, upper bound: 0.0800663
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 163.02
Output dim: 7, lower bound: -0.0802829, upper bound: 0.0800663
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 163.02
Output dim: 7, lower bound: -0.0800703, upper bound: 0.0802839
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 163.02
Output dim: 7, lower bound: -0.0802835, upper bound: 0.0800669

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.0948396, -2.0346169, -3.0965807, -2.0336039, -0.3834203, 0.3848945
1: -3.8309994, -2.3243136, -3.8344245, -2.3243492, -0.5704823, 0.5747613
2: -1.0095346, -0.5155814, -1.0098484, -0.5068595, -0.2043828, 0.1950647
3: -1.8495950, -1.4181634, -1.8508921, -1.4177493, -0.1223908, 0.1233473
4: 0.1466251, 0.3784803, 0.1432148, 0.3821575, -0.1720243, 0.1730665
5: -1.4780428, -1.0904313, -1.4791746, -1.0900942, -0.1305399, 0.1313856
6: -0.6427104, -0.0394320, -0.6430540, -0.0303479, -0.1646771, 0.1544557
7: 0.1227281, 0.7230015, 0.1158198, 0.7236822, -0.4545602, 0.4590510
8: -4.5735412, -3.7448232, -4.5744114, -3.7449880, -0.4104177, 0.4118488
9: -4.7655044, -3.9117961, -4.7693267, -3.9113207, -0.3399622, 0.3433083

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 266
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 3330
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2837
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3213
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 3570
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 3581
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3317
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 3332
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 354

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802353, upper bound: 0.0800458
time: 363.10 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802615, upper bound: 0.0800459
time: 71.15 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.0980599, -2.0332520, -3.0981100, -2.0332556, -0.3856456, 0.3879762
1: -3.8312998, -2.3250926, -3.8313427, -2.3252711, -0.5708832, 0.5709193
2: -1.0103698, -0.5121272, -1.0100884, -0.5121414, -0.1963460, 0.2003894
3: -1.8509314, -1.4168516, -1.8509158, -1.4168460, -0.1249486, 0.1223065
4: 0.1434533, 0.3805670, 0.1436468, 0.3805652, -0.1730883, 0.1736086
5: -1.4793293, -1.0885322, -1.4793289, -1.0885210, -0.1333399, 0.1306432
6: -0.6437280, -0.0373951, -0.6433585, -0.0373957, -0.1547272, 0.1586402
7: 0.1194693, 0.7271605, 0.1194739, 0.7269673, -0.4612623, 0.4582859
8: -4.5751715, -3.7448645, -4.5751886, -3.7448673, -0.4119261, 0.4120697
9: -4.7688971, -3.9083858, -4.7688971, -3.9084129, -0.3462676, 0.3398766

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 266
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 3330
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2837
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3213
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 3570
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 3581
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3317
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 3332
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 354

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0800232, upper bound: 0.0802627
time: 8.30 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0800498, upper bound: 0.0802627
time: 12.08 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.0975280, -2.0332336, -3.0976562, -2.0333538, -0.3853213, 0.3880872
1: -3.8313098, -2.3243423, -3.8346491, -2.3243477, -0.5710247, 0.5748626
2: -1.0116863, -0.5120608, -1.0117545, -0.5062762, -0.2030545, 0.2006599
3: -1.8509985, -1.4168491, -1.8510047, -1.4165916, -0.1252647, 0.1223351
4: 0.1425942, 0.3805770, 0.1424007, 0.3823749, -0.1757905, 0.1744909
5: -1.4793320, -1.0889539, -1.4791806, -1.0890075, -0.1333336, 0.1304032
6: -0.6453149, -0.0373924, -0.6453154, -0.0303443, -0.1630683, 0.1588726
7: 0.1194503, 0.7279785, 0.1157733, 0.7280114, -0.4622898, 0.4610740
8: -4.5746017, -3.7448533, -4.5745673, -3.7449856, -0.4115674, 0.4123125
9: -4.7688971, -3.9082632, -4.7693396, -3.9082513, -0.3463095, 0.3404178

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 266
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3548
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 3500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3571
type: A, layer: 1, pos: 3330
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2837
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 3336
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 3213
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 3570
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 3581
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 3317
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 3332
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3314

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 354

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802369, upper bound: 0.0802636
time: 177.47 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802627, upper bound: 0.0802624
time: 108.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 291.61 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 291.61
Output dim: 7, lower bound: -0.0802353, upper bound: 0.0800458
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 291.61
Output dim: 7, lower bound: -0.0802615, upper bound: 0.0800459
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 291.61
Output dim: 7, lower bound: -0.0800232, upper bound: 0.0802627
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 291.61
Output dim: 7, lower bound: -0.0800498, upper bound: 0.0802627
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 291.61
Output dim: 7, lower bound: -0.0802369, upper bound: 0.0802636
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 291.61
Output dim: 7, lower bound: -0.0802627, upper bound: 0.0802624

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.0943670, -2.0352468, -3.0964696, -2.0341508, -0.3821837, 0.3840365
1: -3.8279462, -2.3292809, -3.8344104, -2.3286736, -0.5627350, 0.5696233
2: -1.0094085, -0.5156635, -1.0096380, -0.5069220, -0.2040047, 0.1928812
3: -1.8502455, -1.4193636, -1.8508869, -1.4188386, -0.1200034, 0.1214943
4: 0.1473234, 0.3780871, 0.1439319, 0.3821479, -0.1713675, 0.1705834
5: -1.4784303, -1.0923352, -1.4791743, -1.0917534, -0.1264449, 0.1284838
6: -0.6405948, -0.0432877, -0.6430416, -0.0336354, -0.1586794, 0.1504453
7: 0.1254652, 0.7213398, 0.1179854, 0.7236639, -0.4518360, 0.4552035
8: -4.5711432, -3.7485971, -4.5743484, -3.7481148, -0.4048436, 0.4080897
9: -4.7647963, -3.9131298, -4.7693219, -3.9124918, -0.3376539, 0.3417567

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 266
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 3548
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 3500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 280
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3330
type: B, layer: 1, pos: 3571
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2837
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 3336
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 3213
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 3570
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3581
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 3317
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 3332
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3314

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 266

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0801217, upper bound: 0.0800445
time: 291.60 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0802362, upper bound: 0.0800458
time: 23.24 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 130.56 + 1922.66 = 2053.23 seconds
