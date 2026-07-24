## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.16436890799999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4758203, 0.4758205)
1: (-10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4666057, 0.4666057)
2: (-8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4128773, 0.4128771)
3: (-10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3284395, 0.3284395)
4: (-9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3763745, 0.3763745)
5: (7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3506265, 0.3506265)
6: (-4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3361690, 0.3361690)
7: (-13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4507301, 0.4507303)
8: (0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2773076, 0.2773077)
9: (-6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4804311, 0.4804308)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.26 + 32.82 = 55.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1660289, upper bound: 0.1660293

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660283, upper bound: 0.1651907
time: 3.11 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660283, upper bound: 0.1660286
time: 3.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.55 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 6.55
Output dim: 5, lower bound: -0.1660283, upper bound: 0.1651907
NS_B2, status: Status.UNKNOWN, split count: 1, time: 6.55
Output dim: 5, lower bound: -0.1660283, upper bound: 0.1660286

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -9.1675816, -8.3232307, -9.1664829, -8.3243504, -0.4733639, 0.4735425
1: -10.3158188, -9.4444799, -10.3149166, -9.4458456, -0.4639802, 0.4641585
2: -8.5259857, -7.8151259, -8.5248184, -7.8177791, -0.4081028, 0.4095926
3: -10.0635166, -9.3887920, -10.0622196, -9.3906841, -0.3246911, 0.3252097
4: -9.9671946, -9.3378487, -9.9653015, -9.3388252, -0.3739338, 0.3729515
5: 7.7539816, 8.3283243, 7.7569065, 8.3272171, -0.3471427, 0.3452764
6: -4.2371969, -3.5506365, -4.2340689, -3.5515242, -0.3325107, 0.3309658
7: -13.7577066, -12.8517342, -13.7521219, -12.8542557, -0.4434586, 0.4404285
8: 0.9234161, 1.3785205, 0.9244347, 1.3759155, -0.2725683, 0.2735033
9: -6.6238995, -6.0457287, -6.6224918, -6.0463724, -0.4779963, 0.4780881

Time for backsubstitution: 20.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 836

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1630099
time: 3.24 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1651857
time: 3.30 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -9.1680298, -8.3224335, -9.1680317, -8.3224335, -0.4749396, 0.4758115
1: -10.3162117, -9.4436102, -10.3162136, -9.4436121, -0.4665947, 0.4670134
2: -8.5261040, -7.8132086, -8.5261040, -7.8132086, -0.4100654, 0.4128768
3: -10.0638485, -9.3873920, -10.0638485, -9.3873940, -0.3259737, 0.3283318
4: -9.9685497, -9.3377151, -9.9685497, -9.3377151, -0.3763752, 0.3738163
5: 7.7518911, 8.3284502, 7.7518911, 8.3284492, -0.3506258, 0.3472961
6: -4.2394104, -3.5505152, -4.2394090, -3.5505166, -0.3357149, 0.3340127
7: -13.7619114, -12.8516083, -13.7619133, -12.8516092, -0.4505293, 0.4416378
8: 0.9233375, 1.3804226, 0.9233375, 1.3804221, -0.2750398, 0.2769003
9: -6.6247845, -6.0456238, -6.6247835, -6.0456257, -0.4825501, 0.4803705

Time for backsubstitution: 20.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 836

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1638479
time: 3.73 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1660236
time: 3.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.76 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 27.76
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1630099
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 27.76
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1651857
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 27.76
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1638479
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 27.76
Output dim: 5, lower bound: -0.1660234, upper bound: 0.1660236

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -9.1654978, -8.3245544, -9.1654873, -8.3243923, -0.4712667, 0.4712210
1: -10.3150797, -9.4449110, -10.3146038, -9.4458590, -0.4632373, 0.4633472
2: -8.5244274, -7.8161378, -8.5240669, -7.8178115, -0.4065232, 0.4078362
3: -10.0631294, -9.3888521, -10.0620975, -9.3907070, -0.3242522, 0.3249772
4: -9.9667158, -9.3382130, -9.9650755, -9.3388948, -0.3736053, 0.3730426
5: 7.7551646, 8.3267632, 7.7569752, 8.3264427, -0.3461404, 0.3446846
6: -4.2358131, -3.5525026, -4.2339802, -3.5524323, -0.3301857, 0.3290100
7: -13.7577152, -12.8518858, -13.7520962, -12.8543072, -0.4428222, 0.4399247
8: 0.9236422, 1.3782363, 0.9244499, 1.3758059, -0.2721934, 0.2731967
9: -6.6211491, -6.0472841, -6.6211495, -6.0463853, -0.4752727, 0.4752092

Time for backsubstitution: 20.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651850, upper bound: 0.1630099
time: 4.25 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651850, upper bound: 0.1630101
time: 3.68 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -9.1675787, -8.3232307, -9.1664810, -8.3243504, -0.4709957, 0.4735415
1: -10.3158169, -9.4444799, -10.3149157, -9.4458456, -0.4633837, 0.4641590
2: -8.5259848, -7.8151278, -8.5248165, -7.8177814, -0.4066746, 0.4095914
3: -10.0635176, -9.3887911, -10.0622196, -9.3906851, -0.3246796, 0.3252608
4: -9.9671926, -9.3378496, -9.9653006, -9.3388262, -0.3745701, 0.3728130
5: 7.7539811, 8.3283243, 7.7569065, 8.3272161, -0.3470953, 0.3450723
6: -4.2371969, -3.5506372, -4.2340698, -3.5515258, -0.3325095, 0.3288765
7: -13.7577066, -12.8519506, -13.7521229, -12.8543558, -0.4434588, 0.4402566
8: 0.9234161, 1.3785195, 0.9244351, 1.3759151, -0.2725687, 0.2733231
9: -6.6238966, -6.0457287, -6.6224918, -6.0463734, -0.4749093, 0.4780867

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651850, upper bound: 0.1651857
time: 3.44 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651848, upper bound: 0.1651857
time: 3.16 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -9.1659470, -8.3237581, -9.1670351, -8.3224735, -0.4728434, 0.4734898
1: -10.3154774, -9.4440451, -10.3158998, -9.4436264, -0.4658518, 0.4662001
2: -8.5245476, -7.8142176, -8.5253525, -7.8132405, -0.4084880, 0.4111185
3: -10.0634613, -9.3874531, -10.0637264, -9.3874168, -0.3255360, 0.3280987
4: -9.9680729, -9.3380785, -9.9683247, -9.3377848, -0.3760457, 0.3739066
5: 7.7530742, 8.3268890, 7.7519608, 8.3276768, -0.3496211, 0.3467031
6: -4.2380266, -3.5523822, -4.2393212, -3.5514228, -0.3333894, 0.3320568
7: -13.7619190, -12.8517647, -13.7618866, -12.8516560, -0.4498928, 0.4411337
8: 0.9235654, 1.3801384, 0.9233546, 1.3803144, -0.2746639, 0.2765946
9: -6.6220341, -6.0471792, -6.6234422, -6.0456371, -0.4798265, 0.4774919

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660111, upper bound: 0.1638420
time: 3.16 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660174, upper bound: 0.1638420
time: 3.33 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -9.1680298, -8.3224325, -9.1680307, -8.3224344, -0.4725723, 0.4758103
1: -10.3162127, -9.4436121, -10.3162136, -9.4436131, -0.4660034, 0.4670131
2: -8.5261021, -7.8132062, -8.5261040, -7.8132095, -0.4086375, 0.4128768
3: -10.0638466, -9.3873920, -10.0638494, -9.3873940, -0.3259624, 0.3283827
4: -9.9685497, -9.3377142, -9.9685497, -9.3377161, -0.3770103, 0.3736777
5: 7.7518907, 8.3284492, 7.7518911, 8.3284502, -0.3505788, 0.3470831
6: -4.2394094, -3.5505178, -4.2394090, -3.5505173, -0.3357142, 0.3319237
7: -13.7619123, -12.8518267, -13.7619133, -12.8517065, -0.4505289, 0.4414654
8: 0.9233389, 1.3804231, 0.9233384, 1.3804221, -0.2750397, 0.2767234
9: -6.6247830, -6.0456238, -6.6247826, -6.0456247, -0.4794621, 0.4803679

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660111, upper bound: 0.1660178
time: 3.05 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660174, upper bound: 0.1660178
time: 3.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 27.81 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1651850, upper bound: 0.1630099
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1651850, upper bound: 0.1630101
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1651850, upper bound: 0.1651857
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1651848, upper bound: 0.1651857
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1660111, upper bound: 0.1638420
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1660174, upper bound: 0.1638420
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1660111, upper bound: 0.1660178
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 27.81
Output dim: 5, lower bound: -0.1660174, upper bound: 0.1660178

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -9.1643972, -8.3256760, -9.1654873, -8.3243923, -0.4702437, 0.4700205
1: -10.3141785, -9.4462776, -10.3146038, -9.4458590, -0.4622359, 0.4621689
2: -8.5232620, -7.8187904, -8.5240669, -7.8178115, -0.4053550, 0.4051754
3: -10.0618305, -9.3907433, -10.0620975, -9.3907070, -0.3228190, 0.3230261
4: -9.9648209, -9.3391895, -9.9650755, -9.3388948, -0.3716910, 0.3721108
5: 7.7580891, 8.3256569, 7.7569752, 8.3264427, -0.3431404, 0.3435473
6: -4.2326860, -3.5533931, -4.2339802, -3.5524323, -0.3273733, 0.3277423
7: -13.7521305, -12.8544130, -13.7520962, -12.8543072, -0.4370770, 0.4372103
8: 0.9246607, 1.3756309, 0.9244499, 1.3758059, -0.2707820, 0.2708495
9: -6.6197419, -6.0479279, -6.6211495, -6.0463853, -0.4743361, 0.4741817

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1647301, upper bound: 0.1630101
time: 4.00 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1647301, upper bound: 0.1630101
time: 3.12 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -9.1659460, -8.3237572, -9.1654873, -8.3243923, -0.4716663, 0.4720683
1: -10.3154764, -9.4440441, -10.3146038, -9.4458590, -0.4635663, 0.4643812
2: -8.5245476, -7.8142195, -8.5240669, -7.8178115, -0.4066763, 0.4097962
3: -10.0634613, -9.3874531, -10.0620975, -9.3907070, -0.3244882, 0.3264308
4: -9.9680710, -9.3380785, -9.9650755, -9.3388948, -0.3750091, 0.3731465
5: 7.7530737, 8.3268890, 7.7569752, 8.3264427, -0.3471055, 0.3448355
6: -4.2380238, -3.5523829, -4.2339802, -3.5524323, -0.3324852, 0.3286462
7: -13.7619171, -12.8517656, -13.7520962, -12.8543072, -0.4464307, 0.4398353
8: 0.9235644, 1.3801394, 0.9244499, 1.3758059, -0.2718992, 0.2754762
9: -6.6220341, -6.0471787, -6.6211495, -6.0463853, -0.4766321, 0.4748173

Time for backsubstitution: 20.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651727, upper bound: 0.1630046
time: 3.02 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651794, upper bound: 0.1630045
time: 3.96 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1664810, -8.3243504, -0.4699740, 0.4723411
1: -10.3149176, -9.4458456, -10.3149157, -9.4458456, -0.4623785, 0.4629798
2: -8.5248175, -7.8177814, -8.5248165, -7.8177814, -0.4055054, 0.4069316
3: -10.0622196, -9.3906851, -10.0622196, -9.3906851, -0.3232471, 0.3233095
4: -9.9652996, -9.3388252, -9.9653006, -9.3388262, -0.3726559, 0.3718822
5: 7.7569065, 8.3272171, 7.7569065, 8.3272161, -0.3440919, 0.3439355
6: -4.2340698, -3.5515270, -4.2340698, -3.5515258, -0.3296969, 0.3276086
7: -13.7521191, -12.8544769, -13.7521229, -12.8543558, -0.4377139, 0.4375412
8: 0.9244351, 1.3759155, 0.9244351, 1.3759151, -0.2711565, 0.2709739
9: -6.6224899, -6.0463729, -6.6224918, -6.0463734, -0.4739733, 0.4770586

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630091, upper bound: 0.1651856
time: 3.44 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630091, upper bound: 0.1651859
time: 3.21 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -9.1680288, -8.3224344, -9.1664810, -8.3243504, -0.4713957, 0.4743881
1: -10.3162136, -9.4436131, -10.3149157, -9.4458456, -0.4637184, 0.4651928
2: -8.5261040, -7.8132076, -8.5248165, -7.8177814, -0.4068291, 0.4115515
3: -10.0638494, -9.3873940, -10.0622196, -9.3906851, -0.3249151, 0.3267143
4: -9.9685488, -9.3377151, -9.9653006, -9.3388262, -0.3759739, 0.3729179
5: 7.7518921, 8.3284492, 7.7569065, 8.3272161, -0.3484461, 0.3452237
6: -4.2394104, -3.5505173, -4.2340698, -3.5515258, -0.3348093, 0.3285131
7: -13.7619133, -12.8518276, -13.7521229, -12.8543558, -0.4468358, 0.4401672
8: 0.9233379, 1.3804221, 0.9244351, 1.3759151, -0.2722740, 0.2756054
9: -6.6247816, -6.0456252, -6.6224918, -6.0463734, -0.4762692, 0.4776943

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651726, upper bound: 0.1651802
time: 3.17 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651792, upper bound: 0.1651802
time: 3.13 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.1659470, -8.3237581, -9.1670256, -8.3225260, -0.4727938, 0.4734826
1: -10.3154774, -9.4440451, -10.3158884, -9.4436321, -0.4658465, 0.4661708
2: -8.5245476, -7.8142176, -8.5253496, -7.8132467, -0.4084785, 0.4110928
3: -10.0634613, -9.3874531, -10.0637197, -9.3874283, -0.3255206, 0.3280948
4: -9.9680729, -9.3380785, -9.9683065, -9.3377895, -0.3760424, 0.3738899
5: 7.7530742, 8.3268890, 7.7519674, 8.3276691, -0.3496156, 0.3466973
6: -4.2380266, -3.5523822, -4.2393131, -3.5514262, -0.3333879, 0.3320470
7: -13.7619190, -12.8517647, -13.7618847, -12.8516598, -0.4498897, 0.4411242
8: 0.9235654, 1.3801384, 0.9233575, 1.3802924, -0.2746443, 0.2765923
9: -6.6220341, -6.0471792, -6.6234331, -6.0456734, -0.4797888, 0.4774842

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1655562, upper bound: 0.1638420
time: 3.07 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1655562, upper bound: 0.1638420
time: 3.07 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.1659479, -8.3237581, -9.1895514, -8.3219585, -0.4758000, 0.4843044
1: -10.3154755, -9.4440470, -10.3184824, -9.4323368, -0.4704206, 0.4708555
2: -8.5245466, -7.8142185, -8.5260601, -7.8042717, -0.4173508, 0.4172137
3: -10.0634604, -9.3874540, -10.0705204, -9.3858414, -0.3300608, 0.3352816
4: -9.9680729, -9.3380785, -9.9693899, -9.3329124, -0.3811307, 0.3747234
5: 7.7530737, 8.3268890, 7.7460861, 8.3290348, -0.3515916, 0.3516312
6: -4.2380266, -3.5523834, -4.2452269, -3.5504115, -0.3342766, 0.3393340
7: -13.7619152, -12.8517618, -13.7623234, -12.8491936, -0.4526746, 0.4415879
8: 0.9235640, 1.3801389, 0.9156713, 1.3814430, -0.2770407, 0.2827536
9: -6.6220350, -6.0471792, -6.6358495, -6.0453405, -0.4816175, 0.4807143

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651795, upper bound: 0.1638421
time: 3.66 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651792, upper bound: 0.1630039
time: 3.08 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.1680298, -8.3224325, -9.1680193, -8.3224840, -0.4725223, 0.4758029
1: -10.3162127, -9.4436121, -10.3161993, -9.4436188, -0.4659986, 0.4669833
2: -8.5261021, -7.8132062, -8.5261021, -7.8132148, -0.4086280, 0.4128513
3: -10.0638466, -9.3873920, -10.0638428, -9.3874054, -0.3259474, 0.3283786
4: -9.9685497, -9.3377142, -9.9685345, -9.3377199, -0.3770072, 0.3736610
5: 7.7518907, 8.3284492, 7.7519002, 8.3284454, -0.3505731, 0.3470774
6: -4.2394094, -3.5505178, -4.2394023, -3.5505199, -0.3357127, 0.3319142
7: -13.7619123, -12.8518267, -13.7619076, -12.8517075, -0.4505267, 0.4414570
8: 0.9233389, 1.3804231, 0.9233422, 1.3804007, -0.2750198, 0.2767212
9: -6.6247830, -6.0456238, -6.6247745, -6.0456610, -0.4794235, 0.4803617

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638352, upper bound: 0.1660175
time: 3.04 seconds

## Relational analysis of NS_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638352, upper bound: 0.1660179
time: 3.18 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1680298, -8.3224335, -9.1905460, -8.3219166, -0.4755275, 0.4858422
1: -10.3162117, -9.4436111, -10.3187923, -9.4323235, -0.4705174, 0.4716594
2: -8.5261021, -7.8132095, -8.5268106, -7.8042402, -0.4175007, 0.4189744
3: -10.0638466, -9.3873940, -10.0706406, -9.3858185, -0.3304875, 0.3355336
4: -9.9685497, -9.3377151, -9.9696150, -9.3328409, -0.3820956, 0.3744946
5: 7.7518902, 8.3284483, 7.7460165, 8.3298101, -0.3525515, 0.3513460
6: -4.2394085, -3.5505171, -4.2453170, -3.5495048, -0.3366013, 0.3392012
7: -13.7619104, -12.8518295, -13.7623510, -12.8492384, -0.4533117, 0.4419200
8: 0.9233389, 1.3804231, 0.9156561, 1.3815503, -0.2774165, 0.2828698
9: -6.6247835, -6.0456243, -6.6371922, -6.0453248, -0.4812527, 0.4826374

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651795, upper bound: 0.1660170
time: 4.61 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651792, upper bound: 0.1651797
time: 2.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.76 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1647301, upper bound: 0.1630101
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1647301, upper bound: 0.1630101
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651727, upper bound: 0.1630046
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651794, upper bound: 0.1630045
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1630091, upper bound: 0.1651856
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1630091, upper bound: 0.1651859
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651726, upper bound: 0.1651802
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651792, upper bound: 0.1651802
NS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1655562, upper bound: 0.1638420
NS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1655562, upper bound: 0.1638420
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651795, upper bound: 0.1638421
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651792, upper bound: 0.1630039
NS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1638352, upper bound: 0.1660175
NS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1638352, upper bound: 0.1660179
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651795, upper bound: 0.1660170
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 5, lower bound: -0.1651792, upper bound: 0.1651797

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -9.1643972, -8.3256760, -9.1643972, -8.3256760, -0.4689531, 0.4689529
1: -10.3141785, -9.4462776, -10.3141785, -9.4462776, -0.4618001, 0.4618003
2: -8.5232620, -7.8187904, -8.5232620, -7.8187904, -0.4043775, 0.4043775
3: -10.0618305, -9.3907433, -10.0618305, -9.3907433, -0.3227803, 0.3227803
4: -9.9648209, -9.3391895, -9.9648209, -9.3391895, -0.3719697, 0.3719697
5: 7.7580891, 8.3256569, 7.7580891, 8.3256569, -0.3429582, 0.3429585
6: -4.2326860, -3.5533931, -4.2326860, -3.5533931, -0.3264457, 0.3264458
7: -13.7521305, -12.8544130, -13.7521305, -12.8544130, -0.4368095, 0.4368095
8: 0.9246607, 1.3756309, 0.9246607, 1.3756309, -0.2706358, 0.2706358
9: -6.6197419, -6.0479279, -6.6197419, -6.0479279, -0.4727960, 0.4727964

Time for backsubstitution: 20.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1647251, upper bound: 0.1629977
time: 4.21 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1647252, upper bound: 0.1630044
time: 3.65 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -9.1643972, -8.3256760, -9.1664801, -8.3243513, -0.4702811, 0.4710119
1: -10.3141785, -9.4462776, -10.3149176, -9.4458456, -0.4622512, 0.4625285
2: -8.5232620, -7.8187904, -8.5248175, -7.8177814, -0.4053810, 0.4059281
3: -10.0618305, -9.3907433, -10.0622196, -9.3906851, -0.3228062, 0.3232074
4: -9.9648209, -9.3391895, -9.9652996, -9.3388252, -0.3718574, 0.3718216
5: 7.7580891, 8.3256569, 7.7569065, 8.3272171, -0.3434875, 0.3435615
6: -4.2326860, -3.5533931, -4.2340698, -3.5515270, -0.3283046, 0.3278370
7: -13.7521305, -12.8544130, -13.7521191, -12.8544769, -0.4373052, 0.4372177
8: 0.9246607, 1.3756309, 0.9244351, 1.3759155, -0.2709193, 0.2708731
9: -6.6197419, -6.0479279, -6.6224899, -6.0463729, -0.4743481, 0.4755049

Time for backsubstitution: 20.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of NS_B1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1647184, upper bound: 0.1630045
time: 3.31 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1647251, upper bound: 0.1630045
time: 3.67 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -9.1659460, -8.3237572, -9.1654768, -8.3244438, -0.4716175, 0.4720614
1: -10.3154764, -9.4440441, -10.3145914, -9.4458637, -0.4635601, 0.4643514
2: -8.5245476, -7.8142195, -8.5240631, -7.8178186, -0.4066668, 0.4097705
3: -10.0634613, -9.3874531, -10.0620937, -9.3907185, -0.3244733, 0.3264266
4: -9.9680710, -9.3380785, -9.9650583, -9.3388996, -0.3750057, 0.3731301
5: 7.7530737, 8.3268890, 7.7569838, 8.3264370, -0.3471000, 0.3448297
6: -4.2380238, -3.5523829, -4.2339735, -3.5524340, -0.3324829, 0.3286362
7: -13.7619171, -12.8517656, -13.7520905, -12.8543091, -0.4464278, 0.4398263
8: 0.9235644, 1.3801394, 0.9244542, 1.3757844, -0.2718790, 0.2754741
9: -6.6220341, -6.0471787, -6.6211405, -6.0464215, -0.4765959, 0.4748106

Time for backsubstitution: 20.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1655563, upper bound: 0.1630042
time: 4.66 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1655563, upper bound: 0.1630046
time: 3.62 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -9.1659489, -8.3237591, -9.1880102, -8.3238726, -0.4746261, 0.4817967
1: -10.3154755, -9.4440460, -10.3172197, -9.4345589, -0.4679506, 0.4687941
2: -8.5245457, -7.8142195, -8.5247822, -7.8088503, -0.4155252, 0.4158947
3: -10.0634594, -9.3874531, -10.0688801, -9.3891354, -0.3290086, 0.3317659
4: -9.9680710, -9.3380775, -9.9661388, -9.3340216, -0.3800988, 0.3739636
5: 7.7530756, 8.3268881, 7.7511339, 8.3278046, -0.3489566, 0.3489354
6: -4.2380261, -3.5523818, -4.2398701, -3.5514200, -0.3333724, 0.3359311
7: -13.7619133, -12.8517628, -13.7525330, -12.8518467, -0.4480250, 0.4402852
8: 0.9235644, 1.3801374, 0.9167695, 1.3769274, -0.2742870, 0.2793039
9: -6.6220341, -6.0471792, -6.6335707, -6.0460815, -0.4784307, 0.4776914

Time for backsubstitution: 20.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 836

## Relational analysis of NS_B1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1655624, upper bound: 0.1630045
time: 3.06 seconds

## Relational analysis of NS_B1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1655624, upper bound: 0.1630045
time: 3.15 seconds

## BFS NS instance: NS_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1643972, -8.3256760, -0.4710121, 0.4702811
1: -10.3149176, -9.4458456, -10.3141785, -9.4462776, -0.4625287, 0.4622512
2: -8.5248175, -7.8177814, -8.5232620, -7.8187904, -0.4059279, 0.4053810
3: -10.0622196, -9.3906851, -10.0618305, -9.3907433, -0.3232074, 0.3228062
4: -9.9652996, -9.3388252, -9.9648209, -9.3391895, -0.3718214, 0.3718574
5: 7.7569065, 8.3272171, 7.7580891, 8.3256569, -0.3435614, 0.3434873
6: -4.2340698, -3.5515270, -4.2326860, -3.5533931, -0.3278369, 0.3283048
7: -13.7521191, -12.8544769, -13.7521305, -12.8544130, -0.4372180, 0.4373050
8: 0.9244351, 1.3759155, 0.9246607, 1.3756309, -0.2708731, 0.2709193
9: -6.6224899, -6.0463729, -6.6197419, -6.0479279, -0.4755049, 0.4743481

Time for backsubstitution: 20.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of NS_B1_A2_A1_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1651733
time: 3.36 seconds

## Relational analysis of NS_B1_A2_A1_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630042, upper bound: 0.1651799
time: 3.15 seconds

## BFS NS instance: NS_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -9.1664801, -8.3243513, -9.1664801, -8.3243513, -0.4699743, 0.4699740
1: -10.3149176, -9.4458456, -10.3149176, -9.4458456, -0.4623775, 0.4623780
2: -8.5248175, -7.8177814, -8.5248175, -7.8177814, -0.4055052, 0.4055052
3: -10.0622196, -9.3906851, -10.0622196, -9.3906851, -0.3233093, 0.3233093
4: -9.9652996, -9.3388252, -9.9652996, -9.3388252, -0.3726559, 0.3726559
5: 7.7569065, 8.3272171, 7.7569065, 8.3272171, -0.3439353, 0.3439351
6: -4.2340698, -3.5515270, -4.2340698, -3.5515270, -0.3276086, 0.3276086
7: -13.7521191, -12.8544769, -13.7521191, -12.8544769, -0.4375415, 0.4375415
8: 0.9244351, 1.3759155, 0.9244351, 1.3759155, -0.2709742, 0.2709739
9: -6.6224899, -6.0463729, -6.6224899, -6.0463729, -0.4739723, 0.4739723

Time for backsubstitution: 24.55 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.09 + 565.99 = 621.08 seconds
