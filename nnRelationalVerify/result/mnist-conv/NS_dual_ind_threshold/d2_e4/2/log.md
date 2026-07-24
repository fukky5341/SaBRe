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
execution time: IAR + RelationalAnalysis = 22.81 + 34.62 = 57.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4604159, upper bound: 0.4604158

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 219
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.54 seconds

### Candidate
type: A, layer: 3, pos: 219

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4500362, upper bound: 0.4509987
time: 3.76 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4510266, upper bound: 0.4510267
time: 3.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.26 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.26
Output dim: 7, lower bound: -0.4500362, upper bound: 0.4509987
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.26
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

Time for backsubstitution: 8.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.48 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4497375, upper bound: 0.4497375
time: 3.91 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4497375, upper bound: 0.4509980
time: 3.93 seconds

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

Time for backsubstitution: 8.45 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 219
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 219

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4510001, upper bound: 0.4497381
time: 3.89 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4510001, upper bound: 0.4510285
time: 3.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 16.60 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 7, lower bound: -0.4497375, upper bound: 0.4497375
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 7, lower bound: -0.4497375, upper bound: 0.4509980
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 7, lower bound: -0.4510001, upper bound: 0.4497381
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 7, lower bound: -0.4510001, upper bound: 0.4510285

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.6269689, -5.8587551, -7.6269689, -5.8587551, -1.1690645, 1.1690645
1: -8.6430454, -6.8176394, -8.6430454, -6.8176394, -1.6559496, 1.6559501
2: -9.3111229, -7.4726076, -9.3111229, -7.4726076, -1.3436363, 1.3436365
3: -10.4216366, -8.8825150, -10.4216366, -8.8825150, -1.0910230, 1.0910230
4: -5.2431226, -3.9832079, -5.2431226, -3.9832079, -0.9861164, 0.9861159
5: -8.4676332, -6.7430820, -8.4676332, -6.7430820, -0.9294410, 0.9294412
6: -12.4402599, -10.3702965, -12.4402599, -10.3702965, -1.2353787, 1.2353787
7: 1.0668323, 2.4814651, 1.0668323, 2.4814651, -1.1003385, 1.1003385
8: -3.2152729, -1.5719337, -3.2152729, -1.5719337, -1.3363647, 1.3363647
9: 0.5344441, 1.8125176, 0.5344441, 1.8125176, -1.2496829, 1.2496829

Time for backsubstitution: 8.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.49 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4442309, upper bound: 0.4452479
time: 4.13 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4459035, upper bound: 0.4456070
time: 3.92 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.6269689, -5.8587551, -7.6266537, -5.8604803, -1.1763525, 1.1788731
1: -8.6430454, -6.8176394, -8.6436214, -6.8073654, -1.6676455, 1.6561718
2: -9.3111229, -7.4726076, -9.3255072, -7.4740252, -1.3546271, 1.3758125
3: -10.4216366, -8.8825150, -10.4222975, -8.8813114, -1.0913815, 1.0905209
4: -5.2431226, -3.9832079, -5.2421842, -3.9818714, -0.9881349, 0.9881835
5: -8.4676332, -6.7430820, -8.4691639, -6.7433462, -0.9291143, 0.9310298
6: -12.4402599, -10.3702965, -12.4411373, -10.3690157, -1.2368169, 1.2363238
7: 1.0668323, 2.4814651, 1.0618598, 2.4903860, -1.1221819, 1.1161408
8: -3.2152729, -1.5719337, -3.2150526, -1.5790739, -1.3426723, 1.3629255
9: 0.5344441, 1.8125176, 0.5332766, 1.8145859, -1.2516885, 1.2502537

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4442309, upper bound: 0.4464197
time: 4.03 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4459035, upper bound: 0.4468844
time: 3.92 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.6266537, -5.8604803, -7.6269689, -5.8587551, -1.1788731, 1.1763525
1: -8.6436214, -6.8073654, -8.6430454, -6.8176394, -1.6561718, 1.6676450
2: -9.3255072, -7.4740252, -9.3111229, -7.4726076, -1.3758123, 1.3546271
3: -10.4222975, -8.8813114, -10.4216366, -8.8825150, -1.0905209, 1.0913815
4: -5.2421842, -3.9818714, -5.2431226, -3.9832079, -0.9881840, 0.9881349
5: -8.4691639, -6.7433462, -8.4676332, -6.7430820, -0.9310293, 0.9291141
6: -12.4411373, -10.3690157, -12.4402599, -10.3702965, -1.2363238, 1.2368169
7: 1.0618598, 2.4903860, 1.0668323, 2.4814651, -1.1161408, 1.1221819
8: -3.2150526, -1.5790739, -3.2152729, -1.5719337, -1.3629255, 1.3426723
9: 0.5332766, 1.8145859, 0.5344441, 1.8125176, -1.2502537, 1.2516885

Time for backsubstitution: 8.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.48 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4451194, upper bound: 0.4452481
time: 3.85 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4468841, upper bound: 0.4456070
time: 3.89 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.6266537, -5.8604803, -7.6266537, -5.8604803, -1.1735802, 1.1735802
1: -8.6436214, -6.8073654, -8.6436214, -6.8073654, -1.6793785, 1.6793780
2: -9.3255072, -7.4740252, -9.3255072, -7.4740252, -1.3728542, 1.3728542
3: -10.4222975, -8.8813114, -10.4222975, -8.8813114, -1.0913682, 1.0913682
4: -5.2421842, -3.9818714, -5.2421842, -3.9818714, -0.9942379, 0.9942379
5: -8.4691639, -6.7433462, -8.4691639, -6.7433462, -0.9338284, 0.9338281
6: -12.4411373, -10.3690157, -12.4411373, -10.3690157, -1.2343559, 1.2343559
7: 1.0618598, 2.4903860, 1.0618598, 2.4903860, -1.1066937, 1.1066933
8: -3.2150526, -1.5790739, -3.2150526, -1.5790739, -1.3593359, 1.3593359
9: 0.5332766, 1.8145859, 0.5332766, 1.8145859, -1.2703404, 1.2703404

Time for backsubstitution: 9.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 1096

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4451195, upper bound: 0.4453166
time: 4.54 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4468842, upper bound: 0.4457800
time: 3.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 18.06 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4442309, upper bound: 0.4452479
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4459035, upper bound: 0.4456070
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4442309, upper bound: 0.4464197
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4459035, upper bound: 0.4468844
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4451194, upper bound: 0.4452481
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4468841, upper bound: 0.4456070
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4451195, upper bound: 0.4453166
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.06
Output dim: 7, lower bound: -0.4468842, upper bound: 0.4457800

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.6244874, -5.8611794, -7.6258516, -5.8598943, -1.1635733, 1.1626668
1: -8.6412687, -6.8290529, -8.6422443, -6.8228240, -1.6500340, 1.6460142
2: -9.3082418, -7.4776349, -9.3098269, -7.4748783, -1.3350916, 1.3358455
3: -10.4199247, -8.8976707, -10.4208460, -8.8894205, -1.0805831, 1.0701423
4: -5.2237549, -3.9872055, -5.2343974, -3.9850097, -0.9589334, 0.9719572
5: -8.4646187, -6.7515688, -8.4662781, -6.7470112, -0.9223375, 0.9187984
6: -12.4383345, -10.3718176, -12.4393921, -10.3709831, -1.2332230, 1.2329826
7: 1.0808222, 2.4790463, 1.0731368, 2.4803777, -1.0879931, 1.0934949
8: -3.2107282, -1.5802507, -3.2132287, -1.5756817, -1.3278780, 1.3254032
9: 0.5367445, 1.8058658, 0.5354793, 1.8095157, -1.2398515, 1.2378554

Time for backsubstitution: 9.28 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.47 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4379241, upper bound: 0.4434546
time: 3.92 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4424429, upper bound: 0.4437399
time: 4.00 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.6241360, -5.8640237, -7.6258283, -5.8615575, -1.1653395, 1.1616235
1: -8.6448498, -6.8239069, -8.6421757, -6.8218555, -1.6523247, 1.6487331
2: -9.3143606, -7.4804897, -9.3100290, -7.4766788, -1.3387506, 1.3364053
3: -10.4242182, -8.9041805, -10.4207382, -8.8943176, -1.1056223, 1.0706830
4: -5.2160344, -3.9763601, -5.2283325, -3.9848425, -0.9615221, 0.9998994
5: -8.4671955, -6.7560339, -8.4663248, -6.7492018, -0.9301844, 0.9188533
6: -12.4393549, -10.3708649, -12.4391937, -10.3707352, -1.2346945, 1.2333946
7: 1.0799792, 2.4876497, 1.0738418, 2.4804921, -1.0876079, 1.0915093
8: -3.2137175, -1.5833945, -3.2131166, -1.5785551, -1.3266783, 1.3264012
9: 0.5355130, 1.8045268, 0.5353353, 1.8076892, -1.2407656, 1.2364879

Time for backsubstitution: 8.48 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4395723, upper bound: 0.4438014
time: 3.81 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4441032, upper bound: 0.4441035
time: 3.91 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.6244874, -5.8611794, -7.6255417, -5.8616228, -1.1708622, 1.1724768
1: -8.6412687, -6.8290529, -8.6428299, -6.8126488, -1.6617622, 1.6462369
2: -9.3082418, -7.4776349, -9.3242397, -7.4762940, -1.3460741, 1.3679996
3: -10.4199247, -8.8976707, -10.4215136, -8.8882170, -1.0809398, 1.0696416
4: -5.2237549, -3.9872055, -5.2334576, -3.9836659, -0.9610271, 0.9740257
5: -8.4646187, -6.7515688, -8.4678230, -6.7472744, -0.9220152, 0.9204171
6: -12.4383345, -10.3718176, -12.4402666, -10.3696947, -1.2346630, 1.2339287
7: 1.0808222, 2.4790463, 1.0681715, 2.4893270, -1.1098490, 1.1092911
8: -3.2107282, -1.5802507, -3.2130017, -1.5828176, -1.3341889, 1.3519850
9: 0.5367445, 1.8058658, 0.5343223, 1.8115819, -1.2418346, 1.2384295

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4379221, upper bound: 0.4443250
time: 3.93 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4424429, upper bound: 0.4446152
time: 3.91 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.6241360, -5.8640237, -7.6255150, -5.8632860, -1.1726284, 1.1714001
1: -8.6448498, -6.8239069, -8.6427603, -6.8116484, -1.6639929, 1.6489563
2: -9.3143606, -7.4804897, -9.3244257, -7.4780951, -1.3497334, 1.3685222
3: -10.4242182, -8.9041805, -10.4214077, -8.8931160, -1.1059818, 1.0701838
4: -5.2160344, -3.9763601, -5.2273927, -3.9835017, -0.9636269, 1.0019684
5: -8.4671955, -6.7560339, -8.4678679, -6.7494631, -0.9298525, 0.9204578
6: -12.4393549, -10.3708649, -12.4400692, -10.3694506, -1.2361345, 1.2343397
7: 1.0799792, 2.4876497, 1.0688789, 2.4894052, -1.1094627, 1.1072936
8: -3.2137175, -1.5833945, -3.2129188, -1.5856934, -1.3329854, 1.3529553
9: 0.5355130, 1.8045268, 0.5341733, 1.8097560, -1.2427430, 1.2370586

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.50 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4395723, upper bound: 0.4447541
time: 4.18 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4441032, upper bound: 0.4450656
time: 3.99 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.6241808, -5.8629084, -7.6258516, -5.8598943, -1.1733842, 1.1699543
1: -8.6418629, -6.8190012, -8.6422443, -6.8228240, -1.6502552, 1.6577749
2: -9.3226929, -7.4790530, -9.3098269, -7.4748783, -1.3672161, 1.3468204
3: -10.4206038, -8.8964663, -10.4208460, -8.8894205, -1.0800829, 1.0704970
4: -5.2228150, -3.9858544, -5.2343974, -3.9850097, -0.9610014, 0.9740429
5: -8.4661827, -6.7518287, -8.4662781, -6.7470112, -0.9239945, 0.9184742
6: -12.4392138, -10.3705177, -12.4393921, -10.3709831, -1.2341690, 1.2344289
7: 1.0758667, 2.4880371, 1.0731368, 2.4803777, -1.1037827, 1.1153684
8: -3.2104897, -1.5873799, -3.2132287, -1.5756817, -1.3544827, 1.3317122
9: 0.5355997, 1.8079309, 0.5354793, 1.8095157, -1.2404304, 1.2398100

Time for backsubstitution: 9.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.55 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4387936, upper bound: 0.4434536
time: 3.95 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4433150, upper bound: 0.4437400
time: 4.06 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.6238441, -5.8657479, -7.6258283, -5.8615575, -1.1751041, 1.1689119
1: -8.6454334, -6.8136616, -8.6421757, -6.8218555, -1.6525478, 1.6604052
2: -9.3288021, -7.4819088, -9.3100290, -7.4766788, -1.3709395, 1.3473730
3: -10.4249363, -8.9029808, -10.4207382, -8.8943176, -1.1051350, 1.0710387
4: -5.2150955, -3.9749966, -5.2283325, -3.9848425, -0.9635906, 1.0018911
5: -8.4687681, -6.7563014, -8.4663248, -6.7492018, -0.9317660, 0.9185281
6: -12.4402332, -10.3695717, -12.4391937, -10.3707352, -1.2356291, 1.2348547
7: 1.0750546, 2.4965417, 1.0738418, 2.4804921, -1.1033664, 1.1134353
8: -3.2135916, -1.5904469, -3.2131166, -1.5785551, -1.3531699, 1.3326707
9: 0.5341997, 1.8065927, 0.5353353, 1.8076892, -1.2414675, 1.2384162

Time for backsubstitution: 9.36 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.59 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4405392, upper bound: 0.4438012
time: 4.79 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4450654, upper bound: 0.4441034
time: 3.98 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.6241808, -5.8629084, -7.6255417, -5.8616228, -1.1681466, 1.1672068
1: -8.6418629, -6.8190012, -8.6428299, -6.8126488, -1.6736040, 1.6695576
2: -9.3226929, -7.4790530, -9.3242397, -7.4762940, -1.3642282, 1.3650169
3: -10.4206038, -8.8964663, -10.4215136, -8.8882170, -1.0809283, 1.0704861
4: -5.2228150, -3.9858544, -5.2334576, -3.9836659, -0.9671412, 0.9801679
5: -8.4661827, -6.7518287, -8.4678230, -6.7472744, -0.9267983, 0.9232187
6: -12.4392138, -10.3705177, -12.4402666, -10.3696947, -1.2322102, 1.2319603
7: 1.0758667, 2.4880371, 1.0681715, 2.4893270, -1.0943580, 1.0998883
8: -3.2104897, -1.5873799, -3.2130017, -1.5828176, -1.3508968, 1.3484006
9: 0.5355997, 1.8079309, 0.5343223, 1.8115819, -1.2605772, 1.2585459

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.43 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4387963, upper bound: 0.4432038
time: 4.06 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4433297, upper bound: 0.4434966
time: 4.03 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.6238441, -5.8657479, -7.6255150, -5.8632860, -1.1698594, 1.1661334
1: -8.6454334, -6.8136616, -8.6427603, -6.8116484, -1.6758327, 1.6721907
2: -9.3288021, -7.4819088, -9.3244257, -7.4780951, -1.3679533, 1.3655453
3: -10.4249363, -8.9029808, -10.4214077, -8.8931160, -1.1059823, 1.0710316
4: -5.2150955, -3.9749966, -5.2273927, -3.9835017, -0.9697275, 1.0079827
5: -8.4687681, -6.7563014, -8.4678679, -6.7494631, -0.9345584, 0.9232352
6: -12.4402332, -10.3695717, -12.4400692, -10.3694506, -1.2336955, 1.2323947
7: 1.0750546, 2.4965417, 1.0688789, 2.4894052, -1.0939727, 1.0979409
8: -3.2135916, -1.5904469, -3.2129188, -1.5856934, -1.3495803, 1.3493915
9: 0.5341997, 1.8065927, 0.5341733, 1.8097560, -1.2615829, 1.2571731

Time for backsubstitution: 9.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 761
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 2236
type: B, layer: 3, pos: 1096
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 2146
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 901
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 1741
type: B, layer: 3, pos: 1236
type: B, layer: 3, pos: 1929
type: B, layer: 3, pos: 2622
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 2803
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1206
type: B, layer: 3, pos: 2250
type: B, layer: 3, pos: 2528
type: B, layer: 3, pos: 1499
type: B, layer: 3, pos: 1734
type: B, layer: 3, pos: 634
type: B, layer: 3, pos: 1779

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 676

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4405738, upper bound: 0.4436198
time: 4.54 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4450896, upper bound: 0.4439451
time: 3.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 17.93 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4379241, upper bound: 0.4434546
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4424429, upper bound: 0.4437399
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4395723, upper bound: 0.4438014
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4441032, upper bound: 0.4441035
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4379221, upper bound: 0.4443250
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4424429, upper bound: 0.4446152
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4395723, upper bound: 0.4447541
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4441032, upper bound: 0.4450656
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4387936, upper bound: 0.4434536
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4433150, upper bound: 0.4437400
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4405392, upper bound: 0.4438012
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4450654, upper bound: 0.4441034
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4387963, upper bound: 0.4432038
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4433297, upper bound: 0.4434966
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4405738, upper bound: 0.4436198
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.93
Output dim: 7, lower bound: -0.4450896, upper bound: 0.4439451

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.6229005, -5.8761144, -7.6209469, -5.8938704, -1.1329956, 1.1408243
1: -8.6402073, -6.8396788, -8.6476650, -6.8430142, -1.6225233, 1.6311913
2: -9.3078890, -7.4885225, -9.3189754, -7.4991932, -1.3165765, 1.3368969
3: -10.4119139, -8.8984852, -10.4043655, -8.8638449, -1.0816407, 1.0473781
4: -5.2235394, -4.0216780, -5.2609801, -4.0452719, -0.8671436, 0.9259036
5: -8.4550915, -6.7516437, -8.4460297, -6.7265778, -0.9081893, 0.8897688
6: -12.4375935, -10.3728571, -12.4378948, -10.3715973, -1.2299595, 1.2284546
7: 1.0814598, 2.4634771, 1.0647178, 2.4520154, -1.0472236, 1.0716004
8: -3.2019200, -1.5805655, -3.1938982, -1.5473895, -1.3123841, 1.2935081
9: 0.5384955, 1.7832441, 0.5252857, 1.7682672, -1.1922460, 1.2154112

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4273019, upper bound: 0.4381001
time: 3.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4344227, upper bound: 0.4398051
time: 4.03 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.6244459, -5.8616657, -7.6256390, -5.8626103, -1.1467953, 1.1579223
1: -8.6412106, -6.8297586, -8.6419125, -6.8259325, -1.6290379, 1.6454964
2: -9.3081970, -7.4780412, -9.3096046, -7.4771366, -1.3277712, 1.3353038
3: -10.4187880, -8.8977470, -10.4144363, -8.8897381, -1.0797262, 1.0526781
4: -5.2237172, -3.9881372, -5.2342572, -3.9897678, -0.8755035, 0.9712734
5: -8.4634895, -6.7515764, -8.4599857, -6.7470632, -0.9219651, 0.9031086
6: -12.4383097, -10.3718739, -12.4392548, -10.3712244, -1.2303247, 1.2313771
7: 1.0809031, 2.4786789, 1.0735002, 2.4796822, -1.0551434, 1.0929623
8: -3.2091780, -1.5802922, -3.2045135, -1.5758801, -1.3273516, 1.2948446
9: 0.5367874, 1.8053179, 0.5356729, 1.8067095, -1.2199678, 1.2325931

Time for backsubstitution: 8.37 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4318674, upper bound: 0.4383078
time: 6.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4389547, upper bound: 0.4400790
time: 4.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.6226044, -5.8789525, -7.6209450, -5.8955326, -1.1347699, 1.1398244
1: -8.6437874, -6.8345346, -8.6475935, -6.8420572, -1.6247997, 1.6339388
2: -9.3139906, -7.4913902, -9.3191824, -7.5010133, -1.3201981, 1.3374233
3: -10.4162159, -8.9050035, -10.4042702, -8.8686905, -1.1067662, 1.0479178
4: -5.2158184, -4.0107660, -5.2549376, -4.0451245, -0.8697529, 0.9538994
5: -8.4576883, -6.7561140, -8.4460955, -6.7287679, -0.9160466, 0.8898165
6: -12.4386177, -10.3719072, -12.4377098, -10.3713512, -1.2314167, 1.2288651
7: 1.0806274, 2.4720831, 1.0653834, 2.4521313, -1.0468216, 1.0696449
8: -3.2049336, -1.5837221, -3.1937947, -1.5502853, -1.3111572, 1.2944870
9: 0.5372386, 1.7819309, 0.5250995, 1.7664533, -1.1931782, 1.2140312

Time for backsubstitution: 9.30 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.54 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4308745, upper bound: 0.4389723
time: 4.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4359252, upper bound: 0.4401690
time: 4.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.6240950, -5.8645077, -7.6256132, -5.8642750, -1.1486578, 1.1568680
1: -8.6447887, -6.8246174, -8.6418438, -6.8249674, -1.6313219, 1.6482148
2: -9.3143139, -7.4808955, -9.3098068, -7.4789414, -1.3314419, 1.3358636
3: -10.4230804, -8.9042568, -10.4143248, -8.8946342, -1.1047697, 1.0532198
4: -5.2159967, -3.9772923, -5.2281933, -3.9896042, -0.8780892, 0.9992166
5: -8.4660645, -6.7560449, -8.4600258, -6.7492509, -0.9298110, 0.9031515
6: -12.4393291, -10.3709192, -12.4390602, -10.3709736, -1.2318344, 1.2317848
7: 1.0800629, 2.4873152, 1.0742049, 2.4798102, -1.0547667, 1.0909743
8: -3.2121596, -1.5834379, -3.2043953, -1.5787568, -1.3261752, 1.2958479
9: 0.5355556, 1.8039722, 0.5355270, 1.8048835, -1.2209120, 1.2312245

Time for backsubstitution: 8.37 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4354179, upper bound: 0.4392492
time: 3.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4404568, upper bound: 0.4404570
time: 4.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.6229005, -5.8761144, -7.6206365, -5.8955994, -1.1402836, 1.1508121
1: -8.6402073, -6.8396788, -8.6482420, -6.8328567, -1.6341810, 1.6314125
2: -9.3078890, -7.4885225, -9.3334093, -7.5006123, -1.3275456, 1.3691301
3: -10.4119139, -8.8984852, -10.4050388, -8.8626461, -1.0820379, 1.0468698
4: -5.2235394, -4.0216780, -5.2600422, -4.0439444, -0.8691347, 0.9279280
5: -8.4550915, -6.7516437, -8.4475718, -6.7268391, -0.9078646, 0.8913922
6: -12.4375935, -10.3728571, -12.4387684, -10.3703070, -1.2314005, 1.2294025
7: 1.0814598, 2.4634771, 1.0598943, 2.4610987, -1.0690403, 1.0872812
8: -3.2019200, -1.5805655, -3.1937151, -1.5545063, -1.3186831, 1.3201170
9: 0.5384955, 1.7832441, 0.5240378, 1.7703280, -1.1941833, 1.2159290

Time for backsubstitution: 9.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4273000, upper bound: 0.4390826
time: 4.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4344227, upper bound: 0.4406733
time: 4.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.6244459, -5.8616657, -7.6253266, -5.8643360, -1.1540837, 1.1677456
1: -8.6412106, -6.8297586, -8.6424990, -6.8157120, -1.6407309, 1.6457186
2: -9.3081970, -7.4780412, -9.3240223, -7.4785547, -1.3387485, 1.3674598
3: -10.4187880, -8.8977470, -10.4151068, -8.8885355, -1.0800853, 1.0521758
4: -5.2237172, -3.9881372, -5.2333188, -3.9884326, -0.8776248, 0.9733419
5: -8.4634895, -6.7515764, -8.4615288, -6.7473245, -0.9216423, 0.9047308
6: -12.4383097, -10.3718739, -12.4401321, -10.3699360, -1.2317657, 1.2323217
7: 1.0809031, 2.4786789, 1.0685592, 2.4886537, -1.0769939, 1.1087480
8: -3.2091780, -1.5802922, -3.2042913, -1.5830197, -1.3336596, 1.3214293
9: 0.5367874, 1.8053179, 0.5345159, 1.8087831, -1.2219529, 1.2331657

Time for backsubstitution: 9.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4318674, upper bound: 0.4392962
time: 6.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4389547, upper bound: 0.4409628
time: 4.44 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.6226044, -5.8789525, -7.6206336, -5.8972607, -1.1420579, 1.1497898
1: -8.6437874, -6.8345346, -8.6481724, -6.8318653, -1.6363983, 1.6341600
2: -9.3139906, -7.4913902, -9.3335962, -7.5024319, -1.3311667, 1.3696184
3: -10.4162159, -8.9050035, -10.4049454, -8.8674936, -1.1071658, 1.0474119
4: -5.2158184, -4.0107660, -5.2539997, -4.0437994, -0.8717320, 0.9559250
5: -8.4576883, -6.7561140, -8.4476318, -6.7290292, -0.9157128, 0.8914204
6: -12.4386177, -10.3719072, -12.4385853, -10.3700638, -1.2328558, 1.2298131
7: 1.0806274, 2.4720831, 1.0605640, 2.4611762, -1.0686107, 1.0853138
8: -3.2049336, -1.5837221, -3.1936436, -1.5574055, -1.3174524, 1.3210621
9: 0.5372386, 1.7819309, 0.5238490, 1.7685156, -1.1951127, 1.2145481

Time for backsubstitution: 9.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4308745, upper bound: 0.4399328
time: 4.17 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4359252, upper bound: 0.4411273
time: 3.97 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.6240950, -5.8645077, -7.6253004, -5.8660021, -1.1559458, 1.1666574
1: -8.6447887, -6.8246174, -8.6424294, -6.8147116, -1.6429558, 1.6484394
2: -9.3143139, -7.4808955, -9.3242054, -7.4803553, -1.3424191, 1.3679814
3: -10.4230804, -8.9042568, -10.4149961, -8.8934345, -1.1051311, 1.0527201
4: -5.2159967, -3.9772923, -5.2272539, -3.9882703, -0.8802199, 1.0012851
5: -8.4660645, -6.7560449, -8.4615612, -6.7495155, -0.9294786, 0.9047596
6: -12.4393291, -10.3709192, -12.4399319, -10.3696899, -1.2332740, 1.2327299
7: 1.0800629, 2.4873152, 1.0692666, 2.4887452, -1.0766139, 1.1067476
8: -3.2121596, -1.5834379, -3.2042036, -1.5858979, -1.3324809, 1.3224058
9: 0.5355556, 1.8039722, 0.5343664, 1.8069580, -1.2228932, 1.2317939

Time for backsubstitution: 9.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4354179, upper bound: 0.4402467
time: 4.02 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4404568, upper bound: 0.4414278
time: 4.18 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.6225924, -5.8778429, -7.6209469, -5.8938704, -1.1429005, 1.1481128
1: -8.6408024, -6.8296280, -8.6476650, -6.8430142, -1.6227446, 1.6429405
2: -9.3223429, -7.4899397, -9.3189754, -7.4991932, -1.3487024, 1.3478703
3: -10.4125834, -8.8972874, -10.4043655, -8.8638449, -1.0811405, 1.0477376
4: -5.2225986, -4.0203223, -5.2609801, -4.0452719, -0.8692117, 0.9279857
5: -8.4566536, -6.7519045, -8.4460297, -6.7265778, -0.9098415, 0.8894441
6: -12.4384727, -10.3715630, -12.4378948, -10.3715973, -1.2309070, 1.2299008
7: 1.0765479, 2.4725642, 1.0647178, 2.4520154, -1.0629921, 1.0934434
8: -3.2016916, -1.5877008, -3.1938982, -1.5473895, -1.3390498, 1.2998147
9: 0.5373473, 1.7853000, 0.5252857, 1.7682672, -1.1928368, 1.2173505

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4282807, upper bound: 0.4381007
time: 4.00 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4353024, upper bound: 0.4398056
time: 4.00 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.6241422, -5.8633938, -7.6256390, -5.8626103, -1.1566372, 1.1652107
1: -8.6418047, -6.8197193, -8.6419125, -6.8259325, -1.6292610, 1.6572590
2: -9.3226461, -7.4794564, -9.3096046, -7.4771366, -1.3598990, 1.3462777
3: -10.4194641, -8.8965416, -10.4144363, -8.8897381, -1.0792260, 1.0530324
4: -5.2227788, -3.9867864, -5.2342572, -3.9897678, -0.8775725, 0.9733596
5: -8.4650583, -6.7518363, -8.4599857, -6.7470632, -0.9236226, 0.9027839
6: -12.4391870, -10.3705740, -12.4392548, -10.3712244, -1.2312717, 1.2328224
7: 1.0759540, 2.4876893, 1.0735002, 2.4796822, -1.0709257, 1.1148343
8: -3.2089386, -1.5874205, -3.2045135, -1.5758801, -1.3539534, 1.3011527
9: 0.5356424, 1.8073897, 0.5356729, 1.8067095, -1.2205286, 1.2345510

Time for backsubstitution: 9.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.50 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4328659, upper bound: 0.4383077
time: 4.53 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4398402, upper bound: 0.4400788
time: 3.92 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.6223135, -5.8806767, -7.6209450, -5.8955326, -1.1446424, 1.1471124
1: -8.6443739, -6.8242912, -8.6475935, -6.8420572, -1.6250229, 1.6455984
2: -9.3284397, -7.4928088, -9.3191824, -7.5010133, -1.3523893, 1.3483906
3: -10.4169264, -8.9038105, -10.4042702, -8.8686905, -1.1062803, 1.0482807
4: -5.2148781, -4.0093951, -5.2549376, -4.0451245, -0.8718219, 0.9558654
5: -8.4592600, -6.7563815, -8.4460955, -6.7287679, -0.9176564, 0.8894908
6: -12.4394970, -10.3706121, -12.4377098, -10.3713512, -1.2323632, 1.2303257
7: 1.0757465, 2.4810691, 1.0653834, 2.4521313, -1.0625591, 1.0915375
8: -3.2048178, -1.5907784, -3.1937947, -1.5502853, -1.3377123, 1.3007550
9: 0.5359210, 1.7839866, 0.5250995, 1.7664533, -1.1938906, 1.2159448

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4318898, upper bound: 0.4389724
time: 4.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4369038, upper bound: 0.4401690
time: 4.06 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.6238050, -5.8662333, -7.6256132, -5.8642750, -1.1584535, 1.1641564
1: -8.6453733, -6.8143806, -8.6418438, -6.8249674, -1.6315470, 1.6598887
2: -9.3287544, -7.4823122, -9.3098068, -7.4789414, -1.3636384, 1.3468318
3: -10.4237957, -8.9030561, -10.4143248, -8.8946342, -1.1042829, 1.0535765
4: -5.2150578, -3.9759274, -5.2281933, -3.9896042, -0.8801587, 1.0012078
5: -8.4676361, -6.7563095, -8.4600258, -6.7492509, -0.9313931, 0.9028244
6: -12.4402084, -10.3696251, -12.4390602, -10.3709736, -1.2327700, 1.2332449
7: 1.0751452, 2.4962254, 1.0742049, 2.4798102, -1.0705180, 1.1128998
8: -3.2120361, -1.5904903, -3.2043953, -1.5787568, -1.3526640, 1.3021173
9: 0.5342424, 1.8060460, 0.5355270, 1.8048835, -1.2215962, 1.2331557

Time for backsubstitution: 9.24 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.54 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4364275, upper bound: 0.4392498
time: 4.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4414271, upper bound: 0.4404572
time: 4.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.6225924, -5.8778429, -7.6206365, -5.8955994, -1.1376095, 1.1454368
1: -8.6408024, -6.8296280, -8.6482420, -6.8328567, -1.6460714, 1.6548448
2: -9.3223429, -7.4899397, -9.3334093, -7.5006123, -1.3457007, 1.3661075
3: -10.4125834, -8.8972874, -10.4050388, -8.8626461, -1.0820179, 1.0477142
4: -5.2225986, -4.0203223, -5.2600422, -4.0439444, -0.8753066, 0.9340932
5: -8.4566536, -6.7519045, -8.4475718, -6.7268391, -0.9126344, 0.8941903
6: -12.4384727, -10.3715630, -12.4387684, -10.3703070, -1.2290282, 1.2274365
7: 1.0765479, 2.4725642, 1.0598943, 2.4610987, -1.0535369, 1.0779324
8: -3.2016916, -1.5877008, -3.1937151, -1.5545063, -1.3354802, 1.3165317
9: 0.5373473, 1.7853000, 0.5240378, 1.7703280, -1.2129869, 1.2360263

Time for backsubstitution: 9.49 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.48 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4283502, upper bound: 0.4380011
time: 4.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4353025, upper bound: 0.4395641
time: 3.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.6241422, -5.8633938, -7.6253266, -5.8643360, -1.1513977, 1.1624699
1: -8.6418047, -6.8197193, -8.6424990, -6.8157120, -1.6525745, 1.6690478
2: -9.3226461, -7.4794564, -9.3240223, -7.4785547, -1.3569069, 1.3644743
3: -10.4194641, -8.8965416, -10.4151068, -8.8885355, -1.0800719, 1.0530202
4: -5.2227788, -3.9867864, -5.2333188, -3.9884326, -0.8837409, 0.9794846
5: -8.4650583, -6.7518363, -8.4615288, -6.7473245, -0.9264235, 0.9075308
6: -12.4391870, -10.3705740, -12.4401321, -10.3699360, -1.2293167, 1.2303548
7: 1.0759540, 2.4876893, 1.0685592, 2.4886537, -1.0614991, 1.0993547
8: -3.2089386, -1.5874205, -3.2042913, -1.5830197, -1.3503685, 1.3178444
9: 0.5356424, 1.8073897, 0.5345159, 1.8087831, -1.2406802, 1.2532849

Time for backsubstitution: 9.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4329818, upper bound: 0.4382487
time: 4.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4398615, upper bound: 0.4398573
time: 3.91 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.6223135, -5.8806767, -7.6206336, -5.8972607, -1.1393456, 1.1444178
1: -8.6443739, -6.8242912, -8.6481724, -6.8318653, -1.6482868, 1.6575065
2: -9.3284397, -7.4928088, -9.3335962, -7.5024319, -1.3493881, 1.3666015
3: -10.4169264, -8.9038105, -10.4049454, -8.8674936, -1.1071587, 1.0482583
4: -5.2148781, -4.0093951, -5.2539997, -4.0437994, -0.8778839, 0.9619379
5: -8.4592600, -6.7563815, -8.4476318, -6.7290292, -0.9204397, 0.8941939
6: -12.4394970, -10.3706121, -12.4385853, -10.3700638, -1.2305107, 1.2278724
7: 1.0757465, 2.4810691, 1.0605640, 2.4611762, -1.0531344, 1.0760140
8: -3.2048178, -1.5907784, -3.1936436, -1.5574055, -1.3341360, 1.3174982
9: 0.5359210, 1.7839866, 0.5238490, 1.7685156, -1.2140145, 1.2346416

Time for backsubstitution: 9.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4320416, upper bound: 0.4388523
time: 4.39 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4369378, upper bound: 0.4400106
time: 4.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.6238050, -5.8662333, -7.6253004, -5.8660021, -1.1532063, 1.1613822
1: -8.6453733, -6.8143806, -8.6424294, -6.8147116, -1.6547985, 1.6716795
2: -9.3287544, -7.4823122, -9.3242054, -7.4803553, -1.3606474, 1.3650022
3: -10.4237957, -8.9030561, -10.4149961, -8.8934345, -1.1051302, 1.0535669
4: -5.2150578, -3.9759274, -5.2272539, -3.9882703, -0.8863223, 1.0072994
5: -8.4676361, -6.7563095, -8.4615612, -6.7495155, -0.9341846, 0.9075348
6: -12.4402084, -10.3696251, -12.4399319, -10.3696899, -1.2308407, 1.2307868
7: 1.0751452, 2.4962254, 1.0692666, 2.4887452, -1.0611215, 1.0974050
8: -3.2120361, -1.5904903, -3.2042036, -1.5858979, -1.3490739, 1.3188415
9: 0.5342424, 1.8060460, 0.5343664, 1.8069580, -1.2417178, 1.2519112

Time for backsubstitution: 9.35 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 761
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2236
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 2146
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 901
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 1741
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 1929
type: A, layer: 3, pos: 2622
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2803
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1206
type: A, layer: 3, pos: 2250
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 1499
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 634
type: A, layer: 3, pos: 1779

Time for candidate selection: 0.53 seconds

### Candidate
type: A, layer: 3, pos: 1978

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4365697, upper bound: 0.4391771
time: 3.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4414556, upper bound: 0.4403162
time: 3.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 17.79 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4273019, upper bound: 0.4381001
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4344227, upper bound: 0.4398051
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4318674, upper bound: 0.4383078
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4389547, upper bound: 0.4400790
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4308745, upper bound: 0.4389723
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4359252, upper bound: 0.4401690
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4354179, upper bound: 0.4392492
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4404568, upper bound: 0.4404570
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4273000, upper bound: 0.4390826
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4344227, upper bound: 0.4406733
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4318674, upper bound: 0.4392962
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4389547, upper bound: 0.4409628
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4308745, upper bound: 0.4399328
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4359252, upper bound: 0.4411273
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4354179, upper bound: 0.4402467
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4404568, upper bound: 0.4414278
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4282807, upper bound: 0.4381007
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4353024, upper bound: 0.4398056
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4328659, upper bound: 0.4383077
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4398402, upper bound: 0.4400788
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4318898, upper bound: 0.4389724
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4369038, upper bound: 0.4401690
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4364275, upper bound: 0.4392498
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4414271, upper bound: 0.4404572
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4283502, upper bound: 0.4380011
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4353025, upper bound: 0.4395641
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4329818, upper bound: 0.4382487
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4398615, upper bound: 0.4398573
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4320416, upper bound: 0.4388523
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4369378, upper bound: 0.4400106
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4365697, upper bound: 0.4391771
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.79
Output dim: 7, lower bound: -0.4414556, upper bound: 0.4403162

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.6251020, -5.8850293, -7.6195183, -5.8977919, -1.1336336, 1.1300998
1: -8.6487522, -6.8687048, -8.6462545, -6.8552604, -1.6291208, 1.6060443
2: -9.2858734, -7.4753375, -9.3092232, -7.4995422, -1.2954443, 1.3475256
3: -10.3607731, -8.9020901, -10.3824387, -8.8666344, -1.0306816, 1.0256977
4: -5.2213092, -4.0423431, -5.2599406, -4.0546660, -0.8596001, 0.9060960
5: -8.4442339, -6.7477026, -8.4408112, -6.7265892, -0.8934484, 0.8841379
6: -12.4198685, -10.3741760, -12.4302063, -10.3721619, -1.2100806, 1.2184768
7: 1.1153173, 2.4685512, 1.0791469, 2.4517035, -1.0118155, 1.0594573
8: -3.1905241, -1.5658493, -3.1883240, -1.5477991, -1.2932220, 1.2912202
9: 0.5601013, 1.7895770, 0.5347846, 1.7677562, -1.1725144, 1.2189121

Time for backsubstitution: 9.22 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.43 + 551.65 = 609.07 seconds
