## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2940516


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5389104, 0.5389103)
1: (-11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4993663, 0.4993664)
2: (6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4197077, 0.4197077)
3: (-4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3537687, 0.3537687)
4: (-12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3765505, 0.3765505)
5: (-13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3536179, 0.3536178)
6: (-10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343599, 0.5343599)
7: (-1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3434693, 0.3434693)
8: (-0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3664252, 0.3664252)
9: (-10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5006785, 0.5006785)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.44 + 35.06 = 58.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3267234, upper bound: 0.3267232

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 675
type: A, layer: 3, pos: 675
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.56 seconds

### Candidate
type: B, layer: 3, pos: 675

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3215697, upper bound: 0.3157591
time: 3.63 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3215697, upper bound: 0.3215703
time: 3.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.77 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 7.77
Output dim: 2, lower bound: -0.3215697, upper bound: 0.3157591
NS_B2, status: Status.UNKNOWN, split count: 1, time: 7.77
Output dim: 2, lower bound: -0.3215697, upper bound: 0.3215703

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -5.9976172, -4.6783876, -5.9843969, -4.6783876, -0.5323088, 0.5188111
1: -11.1173611, -9.8056650, -11.1173601, -9.8143692, -0.4852633, 0.4949231
2: 6.1166925, 7.2846608, 6.1281986, 7.2846136, -0.4132460, 0.4009869
3: -4.7741184, -3.9350276, -4.7735462, -3.9379275, -0.3318352, 0.3449365
4: -12.3435230, -11.2250986, -12.3435221, -11.2320642, -0.3654678, 0.3730470
5: -13.7827301, -12.7531576, -13.7827053, -12.7533398, -0.3515215, 0.3528756
6: -10.9407291, -9.7298355, -10.9382477, -9.7311249, -0.5288534, 0.5291035
7: -1.7099912, -0.7294655, -1.7087009, -0.7294660, -0.3417822, 0.3381132
8: -0.6337786, 0.2915373, -0.6336932, 0.2913942, -0.3657327, 0.3658352
9: -10.0898247, -8.8875408, -10.0884876, -8.8977356, -0.4830256, 0.4910133

Time for backsubstitution: 8.53 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3169894, upper bound: 0.3133620
time: 4.82 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3193731, upper bound: 0.3135634
time: 3.59 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -5.9994783, -4.6783876, -5.9940472, -4.6570978, -0.5715134, 0.5249090
1: -11.1173592, -9.8044310, -11.1307716, -9.8080826, -0.4898131, 0.5211368
2: 6.1160021, 7.2846613, 6.1226683, 7.3025036, -0.4539121, 0.4046220
3: -4.7741356, -3.9437406, -4.7643957, -3.9593542, -0.3251345, 0.3783070
4: -12.3435211, -11.2238083, -12.3546038, -11.2261906, -0.3695075, 0.3929628
5: -13.7827311, -12.7542496, -13.7811842, -12.7561016, -0.3504832, 0.3564539
6: -10.9407167, -9.7297935, -10.9390812, -9.7272129, -0.5331826, 0.5296214
7: -1.7079186, -0.7294660, -1.7040806, -0.7313976, -0.3500289, 0.3374674
8: -0.6337795, 0.2911787, -0.6337004, 0.2905169, -0.3655542, 0.3663979
9: -10.0898628, -8.8863649, -10.1061983, -8.8909626, -0.4878870, 0.5184667

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3169894, upper bound: 0.3191726
time: 4.37 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3193731, upper bound: 0.3193739
time: 3.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 17.35 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 17.35
Output dim: 2, lower bound: -0.3169894, upper bound: 0.3133620
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 17.35
Output dim: 2, lower bound: -0.3193731, upper bound: 0.3135634
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 17.35
Output dim: 2, lower bound: -0.3169894, upper bound: 0.3191726
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 17.35
Output dim: 2, lower bound: -0.3193731, upper bound: 0.3193739

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -5.9742112, -4.6783876, -5.9734144, -4.6783876, -0.5178461, 0.5113554
1: -11.1017761, -9.8060427, -11.1114950, -9.8145161, -0.4795004, 0.4923528
2: 6.1261950, 7.2845850, 6.1317720, 7.2845850, -0.4066215, 0.3981254
3: -4.7710447, -3.9350986, -4.7723880, -3.9379537, -0.3267127, 0.3429511
4: -12.3071384, -11.2254276, -12.3286667, -11.2321949, -0.3260916, 0.3558317
5: -13.7819729, -12.7609453, -13.7824240, -12.7562542, -0.3427938, 0.3314859
6: -10.9380550, -9.7446423, -10.9372492, -9.7376957, -0.5201478, 0.5123182
7: -1.6912506, -0.7297318, -1.7014904, -0.7295675, -0.3309416, 0.3321301
8: -0.6331468, 0.2853804, -0.6334515, 0.2890358, -0.3630183, 0.3608479
9: -10.0733948, -8.8884945, -10.0820045, -8.8980904, -0.4662004, 0.4843496

Time for backsubstitution: 8.53 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3119837, upper bound: 0.3049703
time: 5.17 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049703
time: 3.59 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -5.9942050, -4.6564512, -5.9816847, -4.6783876, -0.5373621, 0.5275989
1: -11.1140308, -9.7891045, -11.1160679, -9.8144112, -0.4852850, 0.4905748
2: 6.1200438, 7.2926602, 6.1302500, 7.2846088, -0.4161909, 0.4004070
3: -4.7707286, -3.9354398, -4.7721810, -3.9379361, -0.3265538, 0.3489528
4: -12.3398514, -11.1882544, -12.3420849, -11.2321005, -0.3422546, 0.4141511
5: -13.7834911, -12.7636395, -13.7826395, -12.7583294, -0.3782060, 0.3298241
6: -10.9734201, -9.7367821, -10.9380226, -9.7337160, -0.5793128, 0.5164848
7: -1.7038217, -0.7140765, -1.7063890, -0.7294874, -0.3366982, 0.3467318
8: -0.6457491, 0.2913375, -0.6336517, 0.2906690, -0.3847650, 0.3707948
9: -10.0855274, -8.8711014, -10.0868874, -8.8978252, -0.4708765, 0.5123410

Time for backsubstitution: 9.13 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3145958, upper bound: 0.3052256
time: 3.79 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110360, upper bound: 0.3052263
time: 3.57 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -5.9760861, -4.6783876, -5.9830551, -4.6570978, -0.5567806, 0.5174366
1: -11.1017771, -9.8048105, -11.1249056, -9.8082304, -0.4840498, 0.5185378
2: 6.1255012, 7.2845874, 6.1262484, 7.3024731, -0.4471034, 0.4017607
3: -4.7710614, -3.9438100, -4.7632370, -3.9593792, -0.3200133, 0.3763263
4: -12.3071375, -11.2241373, -12.3397465, -11.2263222, -0.3301297, 0.3757260
5: -13.7819738, -12.7620392, -13.7808990, -12.7590160, -0.3417556, 0.3350565
6: -10.9380455, -9.7446012, -10.9380798, -9.7337856, -0.5244696, 0.5128389
7: -1.6891801, -0.7297311, -1.6968701, -0.7314966, -0.3391850, 0.3314844
8: -0.6331487, 0.2850227, -0.6334600, 0.2881589, -0.3628343, 0.3614039
9: -10.0734367, -8.8873205, -10.0997162, -8.8913174, -0.4710621, 0.5116608

Time for backsubstitution: 8.51 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3119836, upper bound: 0.3107809
time: 3.94 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3107809
time: 3.58 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -5.9960728, -4.6564512, -5.9913311, -4.6570978, -0.5762002, 0.5336992
1: -11.1140289, -9.7878685, -11.1294813, -9.8081236, -0.4898345, 0.5165749
2: 6.1193533, 7.2926621, 6.1247215, 7.3024964, -0.4566420, 0.4040401
3: -4.7707453, -3.9441519, -4.7630291, -3.9593611, -0.3198616, 0.3823282
4: -12.3398495, -11.1869669, -12.3531666, -11.2262297, -0.3462943, 0.4339094
5: -13.7834902, -12.7647324, -13.7811146, -12.7610912, -0.3771704, 0.3334007
6: -10.9732666, -9.7367420, -10.9388542, -9.7298040, -0.5834119, 0.5170047
7: -1.7017536, -0.7140784, -1.7017694, -0.7314153, -0.3449447, 0.3460876
8: -0.6457505, 0.2909784, -0.6336570, 0.2897935, -0.3845754, 0.3713567
9: -10.0855675, -8.8699493, -10.1045990, -8.8910513, -0.4757387, 0.5395048

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3145958, upper bound: 0.3110361
time: 3.58 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110360, upper bound: 0.3110368
time: 3.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 16.41 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3119837, upper bound: 0.3049703
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049703
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3145958, upper bound: 0.3052256
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3110360, upper bound: 0.3052263
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3119836, upper bound: 0.3107809
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3107809
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3145958, upper bound: 0.3110361
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 16.41
Output dim: 2, lower bound: -0.3110360, upper bound: 0.3110368

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5.9648242, -4.6783876, -5.9734144, -4.6783876, -0.5058618, 0.5113554
1: -11.1017761, -9.8334370, -11.1114950, -9.8145161, -0.4795004, 0.4626707
2: 6.1303911, 7.2841067, 6.1317720, 7.2845850, -0.3935460, 0.3975535
3: -4.7629595, -3.9350979, -4.7723880, -3.9379537, -0.3127962, 0.3429511
4: -12.3067837, -11.2264547, -12.3286667, -11.2321949, -0.3256435, 0.3543409
5: -13.7810020, -12.7617254, -13.7824240, -12.7562542, -0.3415176, 0.3289441
6: -10.9232416, -9.7451172, -10.9372492, -9.7376957, -0.5013947, 0.5111988
7: -1.6911120, -0.7380013, -1.7014904, -0.7295675, -0.3306104, 0.3208280
8: -0.6194816, 0.2853794, -0.6334515, 0.2890358, -0.3403627, 0.3608481
9: -10.0729065, -8.8907757, -10.0820045, -8.8980904, -0.4652157, 0.4804220

Time for backsubstitution: 8.50 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 172

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049698
time: 3.66 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049703
time: 3.69 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -5.9526262, -4.6505003, -5.9647102, -4.6783876, -0.5087316, 0.5616277
1: -11.1878767, -9.8390636, -11.1114950, -9.8273315, -0.5756915, 0.4773773
2: 6.1543336, 7.2967267, 6.1421919, 7.2844138, -0.3891334, 0.4459051
3: -4.7337198, -3.9171648, -4.7586126, -3.9379537, -0.3078697, 0.3920557
4: -12.3065681, -11.2258463, -12.3284473, -11.2326488, -0.3248647, 0.3543668
5: -13.7817049, -12.7696629, -13.7821007, -12.7595291, -0.3482221, 0.3270594
6: -10.9237595, -9.7172203, -10.9304562, -9.7383327, -0.5077715, 0.5338631
7: -1.7173181, -0.7450550, -1.7013195, -0.7369180, -0.3613751, 0.3236705
8: -0.6062093, 0.3198385, -0.6217146, 0.2890348, -0.3442183, 0.4141625
9: -10.0753937, -8.8972073, -10.0813475, -8.9013119, -0.4705863, 0.4780124

Time for backsubstitution: 8.91 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B1_A1_A2_A1

### Relational analysis result of NS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3053861, upper bound: 0.3049696
time: 3.56 seconds

## Relational analysis of NS_B1_A1_A2_A2

### Relational analysis result of NS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3053861, upper bound: 0.3049702
time: 3.29 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -5.9851313, -4.6564512, -5.9816847, -4.6783876, -0.5253160, 0.5275989
1: -11.1140289, -9.8167295, -11.1160679, -9.8144112, -0.4852850, 0.4597806
2: 6.1239810, 7.2921677, 6.1302500, 7.2846088, -0.4031237, 0.3997815
3: -4.7626472, -3.9354410, -4.7721810, -3.9379361, -0.3126210, 0.3489528
4: -12.3394947, -11.1894588, -12.3420849, -11.2321005, -0.3418067, 0.4122862
5: -13.7825041, -12.7643576, -13.7826395, -12.7583294, -0.3766387, 0.3272823
6: -10.9575672, -9.7372351, -10.9380226, -9.7337160, -0.5591404, 0.5153939
7: -1.7037005, -0.7224250, -1.7063890, -0.7294874, -0.3363749, 0.3350983
8: -0.6317558, 0.2913332, -0.6336517, 0.2906690, -0.3630953, 0.3707946
9: -10.0850563, -8.8734016, -10.0868874, -8.8978252, -0.4699156, 0.5084956

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 172

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052255
time: 3.96 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052256
time: 3.12 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -5.9716253, -4.6298127, -5.9729700, -4.6783876, -0.5283215, 0.5752478
1: -11.2001820, -9.8220272, -11.1160650, -9.8272457, -0.5815768, 0.4744373
2: 6.1492243, 7.3035841, 6.1410065, 7.2844300, -0.3987092, 0.4463651
3: -4.7337036, -3.9175034, -4.7583957, -3.9379358, -0.3072366, 0.3994414
4: -12.3392763, -11.1891632, -12.3418789, -11.2325745, -0.3409944, 0.4120076
5: -13.7830439, -12.7728939, -13.7823048, -12.7616844, -0.3834292, 0.3251939
6: -10.9590092, -9.7095642, -10.9311647, -9.7343769, -0.5657072, 0.5378450
7: -1.7304490, -0.7290330, -1.7062113, -0.7368431, -0.3670759, 0.3378438
8: -0.6191144, 0.3258276, -0.6218963, 0.2906680, -0.3668022, 0.4246289
9: -10.0870628, -8.8801107, -10.0862036, -8.9010506, -0.4750822, 0.5060002

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3076171, upper bound: 0.3052266
time: 3.40 seconds

## Relational analysis of NS_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3076171, upper bound: 0.3052261
time: 3.35 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -5.9667015, -4.6783876, -5.9830551, -4.6570978, -0.5448143, 0.5174366
1: -11.1017752, -9.8322020, -11.1249056, -9.8082304, -0.4840498, 0.4890150
2: 6.1297021, 7.2841048, 6.1262484, 7.3024731, -0.4341083, 0.4011899
3: -4.7629757, -3.9438102, -4.7632370, -3.9593792, -0.3060924, 0.3763263
4: -12.3067837, -11.2251663, -12.3397465, -11.2263222, -0.3296815, 0.3743140
5: -13.7810030, -12.7628202, -13.7808990, -12.7590160, -0.3404806, 0.3325181
6: -10.9233036, -9.7450752, -10.9380798, -9.7337856, -0.5057187, 0.5117198
7: -1.6890440, -0.7380009, -1.6968701, -0.7314966, -0.3388543, 0.3201823
8: -0.6194839, 0.2850208, -0.6334600, 0.2881589, -0.3401865, 0.3614039
9: -10.0729465, -8.8895893, -10.0997162, -8.8913174, -0.4700774, 0.5078204

Time for backsubstitution: 8.46 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 172

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3107809
time: 3.84 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3107809
time: 3.18 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -5.9545064, -4.6505003, -5.9743447, -4.6570978, -0.5476468, 0.5677021
1: -11.1878767, -9.8378296, -11.1249075, -9.8210478, -0.5802391, 0.5037345
2: 6.1536427, 7.2967262, 6.1366682, 7.3023014, -0.4297005, 0.4495380
3: -4.7337389, -3.9258766, -4.7494593, -3.9593797, -0.3011658, 0.4254667
4: -12.3065662, -11.2245579, -12.3395290, -11.2267809, -0.3289071, 0.3743607
5: -13.7817039, -12.7707577, -13.7805729, -12.7622900, -0.3471857, 0.3306452
6: -10.9238272, -9.7171783, -10.9313126, -9.7344141, -0.5121140, 0.5343758
7: -1.7152567, -0.7450566, -1.6967018, -0.7388480, -0.3696907, 0.3230244
8: -0.6062083, 0.3194809, -0.6217203, 0.2881575, -0.3440427, 0.4147394
9: -10.0754356, -8.8960180, -10.0990658, -8.8945341, -0.4754486, 0.5054709

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107809
time: 3.24 seconds

## Relational analysis of NS_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3049701
time: 3.29 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -5.9870009, -4.6564512, -5.9913311, -4.6570978, -0.5641673, 0.5336992
1: -11.1140308, -9.8154926, -11.1294813, -9.8081236, -0.4898345, 0.4859203
2: 6.1232896, 7.2921715, 6.1247215, 7.3024964, -0.4436554, 0.4034146
3: -4.7626643, -3.9441531, -4.7630291, -3.9593611, -0.3059242, 0.3823282
4: -12.3394985, -11.1881685, -12.3531666, -11.2262297, -0.3458463, 0.4321074
5: -13.7825031, -12.7654514, -13.7811146, -12.7610912, -0.3756006, 0.3308622
6: -10.9574757, -9.7371960, -10.9388542, -9.7298040, -0.5633411, 0.5159128
7: -1.7016306, -0.7224236, -1.7017694, -0.7314153, -0.3446220, 0.3344541
8: -0.6317539, 0.2909751, -0.6336570, 0.2897935, -0.3629050, 0.3713565
9: -10.0850964, -8.8722429, -10.1045990, -8.8910513, -0.4747776, 0.5357391

Time for backsubstitution: 7.90 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 172

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3110358
time: 4.30 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3110365
time: 3.71 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -5.9734945, -4.6298127, -5.9826150, -4.6570978, -0.5671403, 0.5813410
1: -11.2001829, -9.8207941, -11.1294765, -9.8209610, -0.5861251, 0.5006047
2: 6.1485319, 7.3035841, 6.1354752, 7.3023210, -0.4392458, 0.4499996
3: -4.7337213, -3.9262137, -4.7492399, -3.9593606, -0.3005395, 0.4328543
4: -12.3392754, -11.1878805, -12.3529577, -11.2267036, -0.3450383, 0.4318587
5: -13.7830448, -12.7739868, -13.7807808, -12.7644453, -0.3823906, 0.3287855
6: -10.9589329, -9.7095270, -10.9320230, -9.7304564, -0.5699644, 0.5383584
7: -1.7283864, -0.7290347, -1.7015970, -0.7387712, -0.3753940, 0.3371993
8: -0.6191149, 0.3254671, -0.6219015, 0.2897921, -0.3666133, 0.4252095
9: -10.0871038, -8.8789473, -10.1039143, -8.8942728, -0.4799443, 0.5333176

Time for backsubstitution: 8.52 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B2_A2_A2_A1

### Relational analysis result of NS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3110368
time: 3.62 seconds

## Relational analysis of NS_B2_A2_A2_A2

### Relational analysis result of NS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3052266
time: 3.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 15.88 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049698
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049703
NS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3053861, upper bound: 0.3049696
NS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3053861, upper bound: 0.3049702
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052255
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052256
NS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3076171, upper bound: 0.3052266
NS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3076171, upper bound: 0.3052261
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3107809
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3107809
NS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107809
NS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3049701
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3110358
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3110365
NS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3110368
NS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 15.88
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3052266

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9648242, -4.6783876, -5.9643250, -4.6783876, -0.5058618, 0.4993155
1: -11.1017761, -9.8334370, -11.1114941, -9.8419647, -0.4496453, 0.4626709
2: 6.1303911, 7.2841067, 6.1359777, 7.2840919, -0.3929657, 0.3844465
3: -4.7629595, -3.9350979, -4.7643046, -3.9379547, -0.3127962, 0.3286033
4: -12.3067837, -11.2264547, -12.3283234, -11.2332764, -0.3240703, 0.3539818
5: -13.7810020, -12.7617254, -13.7814102, -12.7570333, -0.3388813, 0.3276293
6: -10.9232416, -9.7451172, -10.9222488, -9.7381744, -0.5002756, 0.4923423
7: -1.6911120, -0.7380013, -1.7013597, -0.7378612, -0.3192973, 0.3205022
8: -0.6194816, 0.2853794, -0.6196761, 0.2890353, -0.3403628, 0.3381600
9: -10.0729065, -8.8907757, -10.0815086, -8.9004011, -0.4612292, 0.4794362

Time for backsubstitution: 8.45 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049702
time: 4.62 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049703
time: 4.65 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9648242, -4.6783876, -5.9508638, -4.6505003, -0.5572102, 0.5094047
1: -11.1017761, -9.8334370, -11.1976147, -9.8476315, -0.4607894, 0.5652484
2: 6.1303911, 7.2841067, 6.1600294, 7.2967091, -0.4416453, 0.3966972
3: -4.7629595, -3.9350979, -4.7350860, -3.9200191, -0.3654137, 0.3337053
4: -12.3067837, -11.2264547, -12.3280573, -11.2326565, -0.3241987, 0.3541434
5: -13.7810020, -12.7617254, -13.7821007, -12.7650700, -0.3398280, 0.3349934
6: -10.9232416, -9.7451172, -10.9225826, -9.7115059, -0.5257773, 0.5040294
7: -1.6911120, -0.7380013, -1.7279162, -0.7449319, -0.3234780, 0.3535523
8: -0.6194816, 0.2853794, -0.6063595, 0.3235903, -0.3963567, 0.3575835
9: -10.0729065, -8.8907757, -10.0837889, -8.9068270, -0.4644865, 0.4847326

Time for backsubstitution: 7.97 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 614
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049703
time: 3.70 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049702
time: 3.71 seconds

## BFS NS instance: NS_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -5.9394178, -4.6505003, -5.9647102, -4.6783876, -0.4953055, 0.5616277
1: -11.1878738, -9.8477783, -11.1114950, -9.8273315, -0.5756907, 0.4675943
2: 6.1658368, 7.2966790, 6.1421919, 7.2844138, -0.3768544, 0.4457901
3: -4.7331343, -3.9200654, -4.7586126, -3.9379537, -0.3062701, 0.3773723
4: -12.3065662, -11.2328053, -12.3284473, -11.2326488, -0.3248644, 0.3467458
5: -13.7816801, -12.7698450, -13.7821007, -12.7595291, -0.3481689, 0.3256327
6: -10.9212799, -9.7185192, -10.9304562, -9.7383327, -0.5050235, 0.5309327
7: -1.7160032, -0.7450566, -1.7013195, -0.7369180, -0.3577211, 0.3236700
8: -0.6061187, 0.3196945, -0.6217146, 0.2890348, -0.3439142, 0.4137677
9: -10.0740204, -8.9074020, -10.0813475, -8.9013119, -0.4678073, 0.4672722

Time for backsubstitution: 8.54 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B1_A1_A2_A1_B1

### Relational analysis result of NS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2991182, upper bound: 0.2978104
time: 4.46 seconds

## Relational analysis of NS_B1_A1_A2_A1_B2

### Relational analysis result of NS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3001856, upper bound: 0.2991719
time: 3.31 seconds

## BFS NS instance: NS_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -5.9490204, -4.6292105, -5.9647102, -4.6783876, -0.5109303, 0.5960445
1: -11.2012882, -9.8415031, -11.1114950, -9.8273315, -0.5978551, 0.4804827
2: 6.1603265, 7.3145695, 6.1421919, 7.2844138, -0.3944524, 0.4803487
3: -4.7239642, -3.9414914, -4.7586126, -3.9379537, -0.3332492, 0.3985921
4: -12.3176470, -11.2269430, -12.3284473, -11.2326488, -0.3420580, 0.3566144
5: -13.7801571, -12.7726068, -13.7821007, -12.7595291, -0.3510775, 0.3277106
6: -10.9221954, -9.7145720, -10.9304562, -9.7383327, -0.5064628, 0.5350980
7: -1.7113922, -0.7469845, -1.7013195, -0.7369180, -0.3619348, 0.3308720
8: -0.6061234, 0.3188162, -0.6217146, 0.2890348, -0.3445590, 0.4138001
9: -10.0917416, -8.9006119, -10.0813475, -8.9013119, -0.4902103, 0.4831328

Time for backsubstitution: 8.87 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B1_A1_A2_A2_B1

### Relational analysis result of NS_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2991182, upper bound: 0.2978106
time: 4.31 seconds

## Relational analysis of NS_B1_A1_A2_A2_B2

### Relational analysis result of NS_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3001856, upper bound: 0.2991712
time: 3.73 seconds

## BFS NS instance: NS_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -5.9851313, -4.6564512, -5.9726090, -4.6783876, -0.5253160, 0.5156472
1: -11.1140289, -9.8167295, -11.1160660, -9.8418856, -0.4554254, 0.4597806
2: 6.1239810, 7.2921677, 6.1342039, 7.2841043, -0.4025394, 0.3869640
3: -4.7626472, -3.9354410, -4.7641001, -3.9379358, -0.3126211, 0.3346217
4: -12.3394947, -11.1894588, -12.3417320, -11.2332125, -0.3402200, 0.4119385
5: -13.7825041, -12.7643576, -13.7816010, -12.7590485, -0.3740725, 0.3259496
6: -10.9575672, -9.7372351, -10.9229259, -9.7341776, -0.5580637, 0.4965195
7: -1.7037005, -0.7224250, -1.7062669, -0.7377949, -0.3250571, 0.3348147
8: -0.6317558, 0.2913332, -0.6198173, 0.2906699, -0.3630950, 0.3481041
9: -10.0850563, -8.8734016, -10.0864067, -8.9001503, -0.4659091, 0.5075395

Time for backsubstitution: 8.97 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B1_A2_A1_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052255
time: 3.92 seconds

## Relational analysis of NS_B1_A2_A1_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052259
time: 3.54 seconds

## BFS NS instance: NS_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -5.9851313, -4.6564512, -5.9591222, -4.6505003, -0.5766644, 0.5250111
1: -11.1140289, -9.8167295, -11.2022123, -9.8475838, -0.4665275, 0.5619099
2: 6.1239810, 7.2921677, 6.1594920, 7.2967205, -0.4512202, 0.3981588
3: -4.7626472, -3.9354410, -4.7350159, -3.9200001, -0.3652441, 0.3396535
4: -12.3394947, -11.1894588, -12.3415108, -11.2326040, -0.3403168, 0.4118127
5: -13.7825041, -12.7643576, -13.7822876, -12.7674894, -0.3751296, 0.3333387
6: -10.9575672, -9.7372351, -10.9231892, -9.7065983, -0.5851092, 0.5081215
7: -1.7037005, -0.7224250, -1.7329237, -0.7448728, -0.3292353, 0.3682198
8: -0.6317558, 0.2913332, -0.6064825, 0.3252816, -0.4200214, 0.3674968
9: -10.0850563, -8.8734016, -10.0884132, -8.9065704, -0.4691713, 0.5128459

Time for backsubstitution: 8.28 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B1_A2_A1_B2_A1

### Relational analysis result of NS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052254
time: 3.87 seconds

## Relational analysis of NS_B1_A2_A1_B2_A2

### Relational analysis result of NS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052258
time: 3.64 seconds

## BFS NS instance: NS_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -5.9584150, -4.6298127, -5.9729700, -4.6783876, -0.5148976, 0.5752478
1: -11.2001820, -9.8307400, -11.1160650, -9.8272457, -0.5815768, 0.4649022
2: 6.1607275, 7.3035364, 6.1410065, 7.2844300, -0.3864468, 0.4462467
3: -4.7331123, -3.9204035, -4.7583957, -3.9379358, -0.3056241, 0.3847631
4: -12.3392763, -11.1961336, -12.3418789, -11.2325745, -0.3409941, 0.4045308
5: -13.7830210, -12.7730751, -13.7823048, -12.7616844, -0.3833786, 0.3237673
6: -10.9565296, -9.7108536, -10.9311647, -9.7343769, -0.5630856, 0.5349184
7: -1.7291603, -0.7290339, -1.7062113, -0.7368431, -0.3634229, 0.3378434
8: -0.6190252, 0.3256826, -0.6218963, 0.2906680, -0.3665317, 0.4242337
9: -10.0856962, -8.8903065, -10.0862036, -8.9010506, -0.4722998, 0.4953030

Time for backsubstitution: 8.79 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B1_A2_A2_A1_B1

### Relational analysis result of NS_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3013316, upper bound: 0.2993504
time: 3.05 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2

### Relational analysis result of NS_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3015666, upper bound: 0.2991715
time: 3.14 seconds

## BFS NS instance: NS_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -5.9680076, -4.6085215, -5.9729700, -4.6783876, -0.5303733, 0.6096647
1: -11.2135963, -9.8244677, -11.1160650, -9.8272457, -0.6037371, 0.4773093
2: 6.1552172, 7.3214288, 6.1410065, 7.2844300, -0.4039916, 0.4808064
3: -4.7239408, -3.9418285, -4.7583957, -3.9379358, -0.3326136, 0.4059777
4: -12.3503571, -11.1902771, -12.3418789, -11.2325745, -0.3581854, 0.4141266
5: -13.7814922, -12.7758389, -13.7823048, -12.7616844, -0.3862571, 0.3258451
6: -10.9573059, -9.7069092, -10.9311647, -9.7343769, -0.5644112, 0.5390857
7: -1.7245560, -0.7309642, -1.7062113, -0.7368431, -0.3676369, 0.3450091
8: -0.6190267, 0.3248043, -0.6218963, 0.2906680, -0.3670645, 0.4242662
9: -10.1034222, -8.8835430, -10.0862036, -8.9010506, -0.4947038, 0.5109617

Time for backsubstitution: 8.29 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B1_A2_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3013316, upper bound: 0.2993505
time: 3.13 seconds

## Relational analysis of NS_B1_A2_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3015666, upper bound: 0.2991711
time: 3.52 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9667015, -4.6783876, -5.9739647, -4.6570978, -0.5448143, 0.5053806
1: -11.1017752, -9.8322020, -11.1249065, -9.8356771, -0.4542317, 0.4890149
2: 6.1297021, 7.2841048, 6.1304579, 7.3019819, -0.4335281, 0.3880829
3: -4.7629757, -3.9438102, -4.7551479, -3.9593797, -0.3060923, 0.3619965
4: -12.3067837, -11.2251663, -12.3394003, -11.2274075, -0.3281239, 0.3739557
5: -13.7810030, -12.7628202, -13.7798834, -12.7597961, -0.3378443, 0.3312076
6: -10.9233036, -9.7450752, -10.9231548, -9.7342577, -0.5046132, 0.4928585
7: -1.6890440, -0.7380009, -1.6967440, -0.7397890, -0.3275536, 0.3198560
8: -0.6194839, 0.2850208, -0.6196837, 0.2881579, -0.3401865, 0.3387607
9: -10.0729465, -8.8895893, -10.0992260, -8.8936138, -0.4660909, 0.5068784

Time for backsubstitution: 8.93 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3107808
time: 3.86 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3049695
time: 8.37 seconds

## BFS NS instance: NS_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9667015, -4.6783876, -5.9604883, -4.6292105, -0.5957661, 0.5154626
1: -11.1017752, -9.8322020, -11.2110291, -9.8413572, -0.4653742, 0.5910959
2: 6.1297021, 7.2841048, 6.1545086, 7.3146005, -0.4818987, 0.4003336
3: -4.7629757, -3.9438102, -4.7259154, -3.9414432, -0.3587101, 0.3672187
4: -12.3067837, -11.2251663, -12.3391380, -11.2267952, -0.3282515, 0.3741163
5: -13.7810030, -12.7628202, -13.7805767, -12.7678337, -0.3387908, 0.3385681
6: -10.9233036, -9.7450752, -10.9234962, -9.7075596, -0.5301826, 0.5045471
7: -1.6890440, -0.7380009, -1.7233121, -0.7468605, -0.3317302, 0.3529374
8: -0.6194839, 0.2850208, -0.6063643, 0.3227124, -0.3961812, 0.3581983
9: -10.0729465, -8.8895893, -10.1015158, -8.9000416, -0.4693493, 0.5123613

Time for backsubstitution: 9.05 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3107808
time: 3.88 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3049697
time: 8.01 seconds

## BFS NS instance: NS_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -5.9394178, -4.6505003, -5.9743447, -4.6570978, -0.5301187, 0.5776759
1: -11.1878738, -9.8477783, -11.1249075, -9.8210478, -0.5881774, 0.4902692
2: 6.1658368, 7.2966790, 6.1366682, 7.3023014, -0.4117254, 0.4634095
3: -4.7331343, -3.9200654, -4.7494593, -3.9593797, -0.3274937, 0.4042410
4: -12.3065662, -11.2328053, -12.3395290, -11.2267809, -0.3346273, 0.3639469
5: -13.7816801, -12.7698450, -13.7805729, -12.7622900, -0.3502033, 0.3285672
6: -10.9212799, -9.7185192, -10.9313126, -9.7344141, -0.5091217, 0.5323830
7: -1.7160032, -0.7450566, -1.6967018, -0.7388480, -0.3649244, 0.3276886
8: -0.6061187, 0.3196945, -0.6217203, 0.2881575, -0.3439459, 0.4143808
9: -10.0740204, -8.9074020, -10.0990658, -8.8945341, -0.4837124, 0.4894929

Time for backsubstitution: 8.98 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 60

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B2_A1_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2967229, upper bound: 0.3036213
time: 5.32 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2977904, upper bound: 0.3049821
time: 4.26 seconds

## BFS NS instance: NS_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -5.9490204, -4.6292105, -5.9743447, -4.6570978, -0.5013685, 0.5677021
1: -11.2012882, -9.8415031, -11.1249075, -9.8210478, -0.5802383, 0.4721667
2: 6.1603265, 7.3145695, 6.1366682, 7.3023014, -0.3805810, 0.4495183
3: -4.7239642, -3.9414914, -4.7494593, -3.9593797, -0.3005285, 0.3716351
4: -12.3176470, -11.2269430, -12.3395290, -11.2267809, -0.3289069, 0.3507966
5: -13.7801571, -12.7726068, -13.7805729, -12.7622900, -0.3471754, 0.3246353
6: -10.9221954, -9.7145720, -10.9313126, -9.7344141, -0.5072989, 0.5332149
7: -1.7113922, -0.7469845, -1.6967018, -0.7388480, -0.3571104, 0.3230286
8: -0.6061234, 0.3188162, -0.6217203, 0.2881575, -0.3439220, 0.4137692
9: -10.0917416, -8.9006119, -10.0990658, -8.8945341, -0.4743352, 0.4737922

Time for backsubstitution: 9.05 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 60

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B2_A1_A2_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2967229, upper bound: 0.2978111
time: 3.98 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2977904, upper bound: 0.2991719
time: 3.31 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -5.9870009, -4.6564512, -5.9822569, -4.6570978, -0.5641673, 0.5217320
1: -11.1140308, -9.8154926, -11.1294794, -9.8356009, -0.4600173, 0.4859202
2: 6.1232896, 7.2921715, 6.1286764, 7.3019962, -0.4430702, 0.3905970
3: -4.7626643, -3.9441531, -4.7549462, -3.9593616, -0.3059242, 0.3680141
4: -12.3394985, -11.1881685, -12.3528099, -11.2273426, -0.3442749, 0.4317605
5: -13.7825031, -12.7654514, -13.7800770, -12.7618122, -0.3730344, 0.3295323
6: -10.9574757, -9.7371960, -10.9238310, -9.7302599, -0.5622778, 0.4970355
7: -1.7016306, -0.7224236, -1.7016480, -0.7397246, -0.3333138, 0.3341700
8: -0.6317539, 0.2909751, -0.6198244, 0.2897911, -0.3629048, 0.3487070
9: -10.0850964, -8.8722429, -10.1041212, -8.8933582, -0.4707711, 0.5348256

Time for backsubstitution: 8.85 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3110365
time: 3.63 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3052258
time: 4.29 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -5.9870009, -4.6564512, -5.9687490, -4.6292105, -0.6151190, 0.5310788
1: -11.1140308, -9.8154926, -11.2156219, -9.8413095, -0.4711175, 0.5875523
2: 6.1232896, 7.2921715, 6.1539655, 7.3146143, -0.4914429, 0.4017916
3: -4.7626643, -3.9441531, -4.7258430, -3.9414244, -0.3585475, 0.3731711
4: -12.3394985, -11.1881685, -12.3525925, -11.2267427, -0.3443707, 0.4316339
5: -13.7825031, -12.7654514, -13.7807636, -12.7702484, -0.3740916, 0.3369183
6: -10.9574757, -9.7371960, -10.9241009, -9.7026520, -0.5893986, 0.5086386
7: -1.7016306, -0.7224236, -1.7283175, -0.7468009, -0.3374891, 0.3676062
8: -0.6317539, 0.2909751, -0.6064868, 0.3244038, -0.4198320, 0.3681142
9: -10.0850964, -8.8722429, -10.1061420, -8.8997850, -0.4740342, 0.5403240

Time for backsubstitution: 8.92 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3110358
time: 3.82 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3052262
time: 4.01 seconds

## BFS NS instance: NS_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -5.9584150, -4.6298127, -5.9826150, -4.6570978, -0.5497110, 0.5914651
1: -11.2001820, -9.8307400, -11.1294765, -9.8209610, -0.5940814, 0.4875833
2: 6.1607275, 7.3035364, 6.1354752, 7.3023210, -0.4213183, 0.4639547
3: -4.7331123, -3.9204035, -4.7492399, -3.9593606, -0.3268576, 0.4116337
4: -12.3392763, -11.1961336, -12.3529577, -11.2267036, -0.3507669, 0.4217355
5: -13.7830210, -12.7730751, -13.7807808, -12.7644453, -0.3854130, 0.3267075
6: -10.9565296, -9.7108536, -10.9320230, -9.7304564, -0.5671904, 0.5363826
7: -1.7291603, -0.7290339, -1.7015970, -0.7387712, -0.3706291, 0.3418637
8: -0.6190252, 0.3256826, -0.6219015, 0.2897921, -0.3665638, 0.4248509
9: -10.0856962, -8.8903065, -10.1039143, -8.8942728, -0.4882731, 0.5175303

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B2_A2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2989399, upper bound: 0.3051613
time: 3.43 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2991713, upper bound: 0.3049827
time: 3.21 seconds

## BFS NS instance: NS_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -5.9680076, -4.6085215, -5.9826150, -4.6570978, -0.5209532, 0.5813410
1: -11.2135963, -9.8244677, -11.1294765, -9.8209610, -0.5861247, 0.4694428
2: 6.1552172, 7.3214288, 6.1354752, 7.3023210, -0.3901740, 0.4499716
3: -4.7239408, -3.9418285, -4.7492399, -3.9593606, -0.2998807, 0.3790264
4: -12.3503571, -11.1902771, -12.3529577, -11.2267036, -0.3450381, 0.4085390
5: -13.7814922, -12.7758389, -13.7807808, -12.7644453, -0.3823820, 0.3227701
6: -10.9573059, -9.7069092, -10.9320230, -9.7304564, -0.5653703, 0.5372030
7: -1.7245560, -0.7309642, -1.7015970, -0.7387712, -0.3628124, 0.3372038
8: -0.6190267, 0.3248043, -0.6219015, 0.2897921, -0.3665050, 0.4242375
9: -10.1034222, -8.8835430, -10.1039143, -8.8942728, -0.4788277, 0.5018277

Time for backsubstitution: 8.93 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_B2_A2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2989399, upper bound: 0.2993503
time: 4.57 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.2991713, upper bound: 0.2991719
time: 3.27 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 17.02 seconds
NS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049702
NS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049703
NS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049703
NS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3085655, upper bound: 0.3049702
NS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2991182, upper bound: 0.2978104
NS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3001856, upper bound: 0.2991719
NS_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2991182, upper bound: 0.2978106
NS_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3001856, upper bound: 0.2991712
NS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052255
NS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052259
NS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052254
NS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3111774, upper bound: 0.3052258
NS_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3013316, upper bound: 0.2993504
NS_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3015666, upper bound: 0.2991715
NS_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3013316, upper bound: 0.2993505
NS_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3015666, upper bound: 0.2991711
NS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3107808
NS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3049695
NS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3107808
NS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3061727, upper bound: 0.3049697
NS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2967229, upper bound: 0.3036213
NS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2977904, upper bound: 0.3049821
NS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2967229, upper bound: 0.2978111
NS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2977904, upper bound: 0.2991719
NS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3110365
NS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3052258
NS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3110358
NS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.3087853, upper bound: 0.3052262
NS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2989399, upper bound: 0.3051613
NS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2991713, upper bound: 0.3049827
NS_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2989399, upper bound: 0.2993503
NS_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.02
Output dim: 2, lower bound: -0.2991713, upper bound: 0.2991719

## BFS NS instance: NS_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.9516091, -4.6783876, -5.9643250, -4.6783876, -0.4924273, 0.4993155
1: -11.1017752, -9.8421383, -11.1114941, -9.8419647, -0.4496450, 0.4528922
2: 6.1418958, 7.2840567, 6.1359777, 7.2840919, -0.3806862, 0.3843197
3: -4.7623873, -3.9379997, -4.7643046, -3.9379547, -0.3111811, 0.3139393
4: -12.3067837, -11.2334223, -12.3283234, -11.2332764, -0.3240701, 0.3463672
5: -13.7809801, -12.7619076, -13.7814102, -12.7570333, -0.3388255, 0.3262027
6: -10.9207563, -9.7464275, -10.9222488, -9.7381744, -0.4975243, 0.4893401
7: -1.6897936, -0.7380016, -1.7013597, -0.7378612, -0.3156228, 0.3205017
8: -0.6193972, 0.2852364, -0.6196761, 0.2890353, -0.3400595, 0.3377645
9: -10.0715466, -8.9009743, -10.0815086, -8.9004011, -0.4584049, 0.4686964

Time for backsubstitution: 8.99 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 900

## Relational analysis of NS_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085668, upper bound: 0.3065034
time: 3.66 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085668, upper bound: 0.3085926
time: 3.68 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9612303, -4.6570978, -5.9643250, -4.6783876, -0.5081260, 0.5341289
1: -11.1151886, -9.8358536, -11.1114941, -9.8419647, -0.4723063, 0.4657652
2: 6.1363835, 7.3019495, 6.1359777, 7.2840919, -0.3982855, 0.4191873
3: -4.7532320, -3.9594254, -4.7643046, -3.9379547, -0.3380397, 0.3351407
4: -12.3178635, -11.2275543, -12.3283234, -11.2332764, -0.3412635, 0.3562039
5: -13.7794552, -12.7646675, -13.7814102, -12.7570333, -0.3417402, 0.3282805
6: -10.9216681, -9.7425156, -10.9222488, -9.7381744, -0.4989643, 0.4934327
7: -1.6851707, -0.7399316, -1.7013597, -0.7378612, -0.3196398, 0.3277075
8: -0.6194034, 0.2843580, -0.6196761, 0.2890353, -0.3406924, 0.3377964
9: -10.0892601, -8.8941879, -10.0815086, -8.9004011, -0.4806184, 0.4845564

Time for backsubstitution: 8.84 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2363
type: B, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: B, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 614
type: B, layer: 3, pos: 614
type: A, layer: 3, pos: 60
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: A, layer: 3, pos: 1849
type: B, layer: 3, pos: 1376
type: A, layer: 3, pos: 1376
type: B, layer: 3, pos: 60

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 900

## Relational analysis of NS_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085668, upper bound: 0.3065034
time: 3.72 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3085668, upper bound: 0.3085926
time: 3.65 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.50 + 545.01 = 603.52 seconds
