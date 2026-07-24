## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.42358272


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6293035, -5.8518434, -7.6293035, -5.8518434, -1.1818094, 1.1818094)
1: (-8.6431360, -6.8047571, -8.6431360, -6.8047571, -1.6936989, 1.6936984)
2: (-9.3191557, -7.4612465, -9.3191557, -7.4612465, -1.3674593, 1.3674593)
3: (-10.4241104, -8.8820829, -10.4241104, -8.8820829, -1.0933785, 1.0933785)
4: (-5.2452884, -3.9792435, -5.2452884, -3.9792435, -0.9980512, 0.9980512)
5: (-8.4706030, -6.7430673, -8.4706030, -6.7430673, -0.9382401, 0.9382403)
6: (-12.4406710, -10.3679934, -12.4406710, -10.3679934, -1.2460556, 1.2460556)
7: (1.0384407, 2.4856093, 1.0384407, 2.4856093, -1.1358390, 1.1358390)
8: (-3.2221670, -1.5668936, -3.2221670, -1.5668936, -1.3539286, 1.3539286)
9: (0.5294220, 1.8146958, 0.5294220, 1.8146958, -1.2844377, 1.2844377)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.25 + 33.73 = 56.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4604159, upper bound: 0.4604158

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 219
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1206
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 634

Time for candidate selection: 0.50 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4500362, upper bound: 0.4509987
time: 3.67 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4510266, upper bound: 0.4510267
time: 3.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.05
Output dim: 7, lower bound: -0.4500362, upper bound: 0.4509987
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.05
Output dim: 7, lower bound: -0.4510266, upper bound: 0.4510267

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.6269689, -5.8587551, -7.6282053, -5.8551636, -1.1751275, 1.1697426
1: -8.6430454, -6.8176394, -8.6430931, -6.8109870, -1.6688862, 1.6622152
2: -9.3111229, -7.4726076, -9.3153248, -7.4666004, -1.3527141, 1.3469644
3: -10.4216366, -8.8825150, -10.4228897, -8.8822880, -1.0912285, 1.0920620
4: -5.2431226, -3.9832079, -5.2442660, -3.9811089, -0.9884162, 0.9901247
5: -8.4676332, -6.7430820, -8.4692039, -6.7430758, -0.9312186, 0.9322214
6: -12.4402599, -10.3702965, -12.4404755, -10.3690777, -1.2393022, 1.2370806
7: 1.0668323, 2.4814651, 1.0522990, 2.4836473, -1.1017308, 1.1173406
8: -3.2152729, -1.5719337, -3.2188768, -1.5692692, -1.3423190, 1.3394938
9: 0.5344441, 1.8125176, 0.5319045, 1.8136516, -1.2594876, 1.2579412

Time for backsubstitution: 9.20 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2803
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 415

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4448445, upper bound: 0.4446779
time: 4.00 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4443879, upper bound: 0.4453200
time: 4.18 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.6266537, -5.8604803, -7.6287303, -5.8559256, -1.1883650, 1.1707282
1: -8.6436214, -6.8073654, -8.6431007, -6.8066568, -1.6752243, 1.6844859
2: -9.3255072, -7.4740252, -9.3182440, -7.4672480, -1.3902211, 1.3535733
3: -10.4222975, -8.8813114, -10.4234219, -8.8821650, -1.0912542, 1.0923381
4: -5.2421842, -3.9818714, -5.2437520, -3.9801662, -0.9907050, 0.9960546
5: -8.4691639, -6.7433462, -8.4698429, -6.7430754, -0.9356222, 0.9316940
6: -12.4411373, -10.3690157, -12.4405937, -10.3684940, -1.2417808, 1.2359543
7: 1.0618598, 2.4903860, 1.0498462, 2.4848571, -1.1031723, 1.1467481
8: -3.2150526, -1.5790739, -3.2212510, -1.5727143, -1.3719563, 1.3421164
9: 0.5332766, 1.8145859, 0.5315106, 1.8145366, -1.2742648, 1.2580171

Time for backsubstitution: 9.20 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1206
type: B, layer: 3, pos: 761
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 1236
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1206

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 415

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4457435, upper bound: 0.4447433
time: 3.96 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4453788, upper bound: 0.4453786
time: 3.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 17.18 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.18
Output dim: 7, lower bound: -0.4448445, upper bound: 0.4446779
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.18
Output dim: 7, lower bound: -0.4443879, upper bound: 0.4453200
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.18
Output dim: 7, lower bound: -0.4457435, upper bound: 0.4447433
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.18
Output dim: 7, lower bound: -0.4453788, upper bound: 0.4453786

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.6242933, -5.8792529, -7.6232471, -5.8936806, -1.1450577, 1.1501861
1: -8.6388359, -6.8237166, -8.6354389, -6.8219757, -1.6395941, 1.6403265
2: -9.3006573, -7.4820423, -9.2961960, -7.4842963, -1.3282099, 1.3204346
3: -10.4166336, -8.8828125, -10.4135094, -8.8828278, -1.0858765, 1.0826626
4: -5.2393198, -3.9872622, -5.2370925, -3.9886568, -0.9804597, 0.9829841
5: -8.4647274, -6.7432880, -8.4638243, -6.7434630, -0.9220266, 0.9212890
6: -12.4315977, -10.3708534, -12.4246922, -10.3701096, -1.2303758, 1.2243538
7: 1.0711124, 2.4799466, 1.0596848, 2.4809484, -1.0839772, 1.0929761
8: -3.2001181, -1.5764937, -3.1938324, -1.5777326, -1.3242149, 1.3206091
9: 0.5352906, 1.8080897, 0.5335774, 1.8057871, -1.2464390, 1.2480865

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1206

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1206

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4119436, upper bound: 0.4202222
time: 3.89 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3962416, upper bound: 0.3954262
time: 3.54 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.6231766, -5.8728447, -7.6361051, -5.8797035, -1.1412287, 1.1573739
1: -8.6361284, -6.8200550, -8.6324501, -6.8119192, -1.6352806, 1.6536732
2: -9.3058014, -7.4972367, -9.2995224, -7.5128822, -1.3005459, 1.2962933
3: -10.4168987, -8.8829784, -10.4118147, -8.8822041, -1.0870686, 1.0814185
4: -5.2402167, -3.9879017, -5.2395225, -3.9878032, -0.9714260, 0.9803557
5: -8.4644289, -6.7435565, -8.4617653, -6.7439518, -0.9219718, 0.9204478
6: -12.4299603, -10.3709450, -12.4229860, -10.3713894, -1.2198830, 1.2236753
7: 1.0716248, 2.4803262, 1.0601742, 2.4838748, -1.0925407, 1.0796175
8: -3.2015877, -1.5761538, -3.2007766, -1.5551844, -1.3111887, 1.3148727
9: 0.5354987, 1.8087296, 0.5328178, 1.8062277, -1.2468357, 1.2524157

Time for backsubstitution: 8.34 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2236
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 1741
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1206

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1206

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4079013, upper bound: 0.4201901
time: 3.47 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3944878, upper bound: 0.3953578
time: 3.58 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.6239829, -5.8809853, -7.6238012, -5.8942852, -1.1588492, 1.1514764
1: -8.6396475, -6.8131962, -8.6354465, -6.8174748, -1.6460991, 1.6620312
2: -9.3156872, -7.4831142, -9.2994881, -7.4840097, -1.3673334, 1.3273702
3: -10.4172468, -8.8815899, -10.4139576, -8.8826914, -1.0859051, 1.0829258
4: -5.2383881, -3.9859171, -5.2365832, -3.9876642, -0.9826488, 0.9891438
5: -8.4662762, -6.7435498, -8.4644985, -6.7434635, -0.9264393, 0.9206438
6: -12.4324837, -10.3695641, -12.4248123, -10.3694973, -1.2328620, 1.2234106
7: 1.0664401, 2.4890664, 1.0574212, 2.4822731, -1.0852036, 1.1223044
8: -3.1995149, -1.5838199, -3.1964383, -1.5811787, -1.3535829, 1.3233070
9: 0.5341281, 1.8102827, 0.5331098, 1.8067303, -1.2611132, 1.2482185

Time for backsubstitution: 8.41 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2236
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1734
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 634
type: B, layer: 3, pos: 1206

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1206

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4116328, upper bound: 0.4218085
time: 3.76 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4002442, upper bound: 0.3988861
time: 3.53 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.6228161, -5.8738155, -7.6365318, -5.8801107, -1.1557040, 1.1587305
1: -8.6371746, -6.8110204, -8.6324558, -6.8082986, -1.6408701, 1.6753998
2: -9.3195419, -7.4986486, -9.3021641, -7.5135217, -1.3379397, 1.3024063
3: -10.4175320, -8.8818102, -10.4125137, -8.8821068, -1.0870647, 1.0816369
4: -5.2392602, -3.9866483, -5.2390013, -3.9870045, -0.9733973, 0.9863510
5: -8.4659090, -6.7438173, -8.4623356, -6.7439518, -0.9266553, 0.9196982
6: -12.4307852, -10.3697128, -12.4230881, -10.3708878, -1.2218828, 1.2224398
7: 1.0670085, 2.4890618, 1.0579090, 2.4848905, -1.0940928, 1.1082191
8: -3.2002902, -1.5834270, -3.2030478, -1.5590572, -1.3409066, 1.3173008
9: 0.5344052, 1.8104584, 0.5325277, 1.8069625, -1.2616081, 1.2518725

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 761
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 1096
type: A, layer: 3, pos: 1096
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 227
type: B, layer: 3, pos: 2146
type: A, layer: 3, pos: 2146
type: B, layer: 3, pos: 2622
type: A, layer: 3, pos: 1929
type: B, layer: 3, pos: 1929
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2622
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1741
type: A, layer: 3, pos: 95
type: B, layer: 3, pos: 95
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 901
type: A, layer: 3, pos: 901
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 2803
type: B, layer: 3, pos: 411
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1236
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 604
type: A, layer: 3, pos: 604
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1734
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1499
type: B, layer: 3, pos: 1499
type: A, layer: 3, pos: 2250
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 634
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1206

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1206

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4094694, upper bound: 0.4218282
time: 3.78 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3988435, upper bound: 0.3988450
time: 3.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 16.12 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.4119436, upper bound: 0.4202222
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.3962416, upper bound: 0.3954262
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.4079013, upper bound: 0.4201901
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.3944878, upper bound: 0.3953578
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.4116328, upper bound: 0.4218085
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.4002442, upper bound: 0.3988861
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.4094694, upper bound: 0.4218282
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 16.12
Output dim: 7, lower bound: -0.3988435, upper bound: 0.3988450

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 56.98 + 106.98 = 163.96 seconds
