## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.635617521


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5792060, 1.5792060)
1: (-6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4270267, 1.4270267)
2: (-8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3698587, 1.3698590)
3: (-9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3052292, 1.3052292)
4: (-4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5216794, 1.5216794)
5: (-5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4439821, 1.4439824)
6: (-13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0481853, 2.0481853)
7: (3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1529670, 1.1529672)
8: (-4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5045223, 1.5045223)
9: (-1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5415206, 1.5415206)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.15 + 37.22 = 59.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.6420365, upper bound: 0.6420360

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405173, upper bound: 0.6420341
time: 3.65 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420326, upper bound: 0.6420321
time: 3.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.61 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.61
Output dim: 7, lower bound: -0.6405173, upper bound: 0.6420341
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.61
Output dim: 7, lower bound: -0.6420326, upper bound: 0.6420321

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.7995949, -6.7005539, -8.8144798, -6.6996937, -1.5445008, 1.5576982
1: -6.3827133, -4.6866484, -6.3854980, -4.6786838, -1.3914986, 1.3863108
2: -8.4177094, -6.8326960, -8.4206667, -6.8298345, -1.3564548, 1.3564878
3: -9.7877798, -7.9882631, -9.7891169, -7.9817119, -1.2959757, 1.2902114
4: -4.6696796, -3.0276978, -4.6722927, -3.0155001, -1.4973073, 1.4909616
5: -5.0489936, -3.3568306, -5.0593328, -3.3549399, -1.4118671, 1.4183505
6: -13.3279057, -11.1606426, -13.3339672, -11.1390591, -2.0224552, 2.0051622
7: 3.6004035, 4.8243914, 3.5956483, 4.8255525, -1.1437716, 1.1473680
8: -4.0765510, -2.0235472, -4.0794468, -2.0122800, -1.4905543, 1.4817667
9: -1.9144611, -0.2552704, -1.9234422, -0.2536776, -1.5228972, 1.5306220

Time for backsubstitution: 20.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404991, upper bound: 0.6397176
time: 5.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405158, upper bound: 0.6420323
time: 3.61 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.8299942, -6.6611004, -8.8265467, -6.6989460, -1.5740986, 1.6039009
1: -6.4108148, -4.6653576, -6.3877411, -4.6722364, -1.4332790, 1.4511693
2: -8.4339132, -6.8236566, -8.4230814, -6.8273768, -1.3767791, 1.3759618
3: -9.8074360, -7.9711056, -9.7901926, -7.9756527, -1.3233321, 1.3082738
4: -4.6982703, -2.9988279, -4.6745548, -3.0056534, -1.5368700, 1.5277772
5: -5.0721436, -3.3324804, -5.0677176, -3.3530443, -1.4490790, 1.4544401
6: -13.3942900, -11.1162834, -13.3388329, -11.1216307, -2.1047997, 2.0487399
7: 3.5799646, 4.8459330, 3.5913973, 4.8264761, -1.1639650, 1.1735651
8: -4.1099014, -2.0007520, -4.0817299, -2.0028381, -1.5331044, 1.5047121
9: -1.9441811, -0.2351732, -1.9310335, -0.2524244, -1.5522714, 1.5589108

Time for backsubstitution: 19.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420144, upper bound: 0.6397171
time: 3.38 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420311, upper bound: 0.6420307
time: 3.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 26.86 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.86
Output dim: 7, lower bound: -0.6404991, upper bound: 0.6397176
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.86
Output dim: 7, lower bound: -0.6405158, upper bound: 0.6420323
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.86
Output dim: 7, lower bound: -0.6420144, upper bound: 0.6397171
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.86
Output dim: 7, lower bound: -0.6420311, upper bound: 0.6420307

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8.7994766, -6.7021656, -8.8137636, -6.7091045, -1.5348301, 1.5550470
1: -6.3826027, -4.6867695, -6.3848524, -4.6793895, -1.3907495, 1.3855369
2: -8.4175463, -6.8337278, -8.4197111, -6.8358788, -1.3478436, 1.3511887
3: -9.7877092, -7.9897079, -9.7886925, -7.9901886, -1.2859313, 1.2855494
4: -4.6691713, -3.0277996, -4.6693144, -3.0161109, -1.4937615, 1.4868269
5: -5.0487881, -3.3589211, -5.0581236, -3.3672040, -1.3993387, 1.4146895
6: -13.3263330, -11.1607084, -13.3247480, -11.1394424, -2.0204406, 1.9958057
7: 3.6010606, 4.8242936, 3.5994878, 4.8249640, -1.1426234, 1.1433792
8: -4.0764408, -2.0244946, -4.0787821, -2.0177984, -1.4827557, 1.4778314
9: -1.9136140, -0.2553366, -1.9184675, -0.2540772, -1.5194101, 1.5241256

Time for backsubstitution: 20.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367101, upper bound: 0.6383796
time: 3.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404958, upper bound: 0.6397142
time: 3.39 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.7995977, -6.7005625, -8.8499660, -6.6982899, -1.5439839, 1.5847669
1: -6.3827105, -4.6866488, -6.3898792, -4.6766300, -1.3934579, 1.3912258
2: -8.4177094, -6.8327003, -8.4402933, -6.8273826, -1.3655620, 1.3728590
3: -9.7877798, -7.9882751, -9.8153734, -7.9797993, -1.3014042, 1.3140602
4: -4.6696768, -3.0276983, -4.6737785, -3.0121303, -1.4979692, 1.4961057
5: -5.0489931, -3.3568399, -5.1045532, -3.3541477, -1.4109540, 1.4465880
6: -13.3279018, -11.1606436, -13.3386745, -11.1063175, -2.0550566, 2.0076323
7: 3.6004074, 4.8243914, 3.5940039, 4.8368979, -1.1551981, 1.1483097
8: -4.0765491, -2.0235562, -4.1001372, -2.0098181, -1.4990373, 1.4998519
9: -1.9144589, -0.2552700, -1.9255691, -0.2413156, -1.5330911, 1.5364962

Time for backsubstitution: 20.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367267, upper bound: 0.6406957
time: 3.60 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405125, upper bound: 0.6420293
time: 3.67 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.8298750, -6.6627121, -8.8258286, -6.7083540, -1.5644293, 1.6009128
1: -6.4107075, -4.6654797, -6.3870978, -4.6729345, -1.4325309, 1.4503925
2: -8.4337511, -6.8246875, -8.4221325, -6.8334184, -1.3681517, 1.3706422
3: -9.8073654, -7.9725518, -9.7897625, -7.9841318, -1.3132882, 1.3036113
4: -4.6977663, -2.9989295, -4.6715794, -3.0062640, -1.5333271, 1.5236387
5: -5.0719385, -3.3345690, -5.0665154, -3.3653061, -1.4365482, 1.4507830
6: -13.3927126, -11.1163483, -13.3296194, -11.1220188, -2.1025395, 2.0393887
7: 3.5806203, 4.8458323, 3.5952332, 4.8258848, -1.1628156, 1.1695774
8: -4.1097870, -2.0017028, -4.0810552, -2.0083523, -1.5252218, 1.5007739
9: -1.9433322, -0.2352425, -1.9260612, -0.2528256, -1.5487866, 1.5524175

Time for backsubstitution: 19.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382424, upper bound: 0.6383780
time: 3.38 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420111, upper bound: 0.6397141
time: 3.53 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.8299952, -6.6611128, -8.8620472, -6.6975541, -1.5735779, 1.6063282
1: -6.4108138, -4.6653590, -6.3921213, -4.6701899, -1.4351997, 1.4560852
2: -8.4339104, -6.8236609, -8.4427032, -6.8249087, -1.3859096, 1.3906951
3: -9.8074360, -7.9711180, -9.8164463, -7.9737344, -1.3287458, 1.3321273
4: -4.6982665, -2.9988291, -4.6760612, -3.0022597, -1.5375462, 1.5329127
5: -5.0721436, -3.3324907, -5.1129694, -3.3522418, -1.4481664, 1.4673312
6: -13.3942833, -11.1162853, -13.3434563, -11.0888653, -2.1061897, 2.0511546
7: 3.5799663, 4.8459315, 3.5897446, 4.8378196, -1.1753917, 1.1745145
8: -4.1099005, -2.0007591, -4.1024294, -2.0003786, -1.5343637, 1.5228596
9: -1.9441793, -0.2351749, -1.9331706, -0.2400621, -1.5624700, 1.5647864

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382591, upper bound: 0.6406940
time: 3.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420277, upper bound: 0.6420276
time: 3.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.45 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6367101, upper bound: 0.6383796
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6404958, upper bound: 0.6397142
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6367267, upper bound: 0.6406957
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6405125, upper bound: 0.6420293
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6382424, upper bound: 0.6383780
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6420111, upper bound: 0.6397141
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6382591, upper bound: 0.6406940
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 7, lower bound: -0.6420277, upper bound: 0.6420276

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.7812729, -6.7066903, -8.8060331, -6.7102151, -1.5126748, 1.5405211
1: -6.3780766, -4.6916823, -6.3837643, -4.6813240, -1.3828950, 1.3792710
2: -8.3972359, -6.8407459, -8.4108839, -6.8377557, -1.3263941, 1.3355119
3: -9.7622290, -8.0013275, -9.7775526, -7.9934263, -1.2561934, 1.2617226
4: -4.6504025, -3.0343227, -4.6611557, -3.0181038, -1.4731236, 1.4681196
5: -5.0146055, -3.3740706, -5.0431190, -3.3701553, -1.3612647, 1.3821864
6: -13.3094540, -11.1653547, -13.3180676, -11.1402779, -2.0008850, 1.9785161
7: 3.6235583, 4.8215151, 3.6090326, 4.8237972, -1.1177931, 1.1302674
8: -4.0673990, -2.0506301, -4.0772047, -2.0291462, -1.4600730, 1.4511833
9: -1.9034801, -0.2649317, -1.9147081, -0.2581886, -1.5041900, 1.5098264

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367101, upper bound: 0.6368384
time: 3.77 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367101, upper bound: 0.6383796
time: 4.15 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.7994719, -6.7021661, -8.8137627, -6.7091074, -1.5302811, 1.5550442
1: -6.3826013, -4.6867723, -6.3848519, -4.6793900, -1.3897777, 1.3870277
2: -8.4175358, -6.8337283, -8.4197092, -6.8358788, -1.3428025, 1.3502212
3: -9.7877026, -7.9897089, -9.7886896, -7.9901896, -1.2746704, 1.2855458
4: -4.6691656, -3.0278020, -4.6693130, -3.0161107, -1.4908171, 1.4840498
5: -5.0487800, -3.3589232, -5.0581231, -3.3672042, -1.3723946, 1.4134488
6: -13.3263273, -11.1607065, -13.3247471, -11.1394424, -2.0196939, 2.0084901
7: 3.6010675, 4.8242927, 3.5994904, 4.8249636, -1.1426187, 1.1434460
8: -4.0764389, -2.0245070, -4.0787835, -2.0178003, -1.4814105, 1.4618444
9: -1.9136101, -0.2553413, -1.9184664, -0.2540798, -1.5194054, 1.5213866

Time for backsubstitution: 20.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6391441, upper bound: 0.6359444
time: 3.55 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6391441, upper bound: 0.6397167
time: 3.60 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.7813921, -6.7050924, -8.8422432, -6.6994085, -1.5218220, 1.5666628
1: -6.3781881, -4.6915593, -6.3887930, -4.6785650, -1.3856072, 1.3849592
2: -8.3973999, -6.8397183, -8.4314861, -6.8292856, -1.3440962, 1.3572040
3: -9.7622948, -7.9998951, -9.8042498, -7.9830618, -1.2716811, 1.2902522
4: -4.6509018, -3.0342195, -4.6655855, -3.0141180, -1.4773340, 1.4774246
5: -5.0148039, -3.3719912, -5.0895615, -3.3571067, -1.3728676, 1.4038699
6: -13.3110209, -11.1652918, -13.3319979, -11.1071482, -2.0355024, 1.9903717
7: 3.6229072, 4.8216128, 3.6035562, 4.8357339, -1.1303666, 1.1351924
8: -4.0675092, -2.0496888, -4.0985575, -2.0211716, -1.4763579, 1.4730468
9: -1.9043262, -0.2648637, -1.9217802, -0.2454181, -1.5178823, 1.5222194

Time for backsubstitution: 20.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367267, upper bound: 0.6391634
time: 3.93 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367267, upper bound: 0.6406957
time: 3.63 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.7995901, -6.7005653, -8.8499641, -6.6982923, -1.5394359, 1.5834613
1: -6.3827124, -4.6866508, -6.3898802, -4.6766291, -1.3924861, 1.3926857
2: -8.4176989, -6.8327007, -8.4402905, -6.8273840, -1.3605032, 1.3718910
3: -9.7877731, -7.9882760, -9.8153715, -7.9797993, -1.2901430, 1.3140562
4: -4.6696711, -3.0276990, -4.6737766, -3.0121310, -1.4949999, 1.4933305
5: -5.0489860, -3.3568437, -5.1045504, -3.3541489, -1.3839998, 1.4378970
6: -13.3278961, -11.1606436, -13.3386726, -11.1063166, -2.0543098, 2.0203171
7: 3.6004128, 4.8243914, 3.5940058, 4.8368969, -1.1551931, 1.1483757
8: -4.0765505, -2.0235639, -4.1001382, -2.0098219, -1.4976902, 1.4834409
9: -1.9144543, -0.2552745, -1.9255679, -0.2413152, -1.5330868, 1.5337567

Time for backsubstitution: 20.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6391621, upper bound: 0.6382590
time: 3.78 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6391621, upper bound: 0.6420293
time: 3.89 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.8115120, -6.6673427, -8.8181114, -6.7094660, -1.5420170, 1.5825865
1: -6.4061041, -4.6704025, -6.3860097, -4.6748548, -1.4246283, 1.4439628
2: -8.4134331, -6.8319325, -8.4133635, -6.8353052, -1.3465705, 1.3546133
3: -9.7817488, -7.9855280, -9.7786274, -7.9873958, -1.2834466, 1.2779107
4: -4.6787696, -3.0053127, -4.6634312, -3.0082517, -1.5124359, 1.5047469
5: -5.0377107, -3.3502743, -5.0515127, -3.3682668, -1.3983665, 1.4150283
6: -13.3758078, -11.1210785, -13.3229475, -11.1228514, -2.0828829, 2.0220513
7: 3.6035066, 4.8429976, 3.6047206, 4.8247185, -1.1376495, 1.1565259
8: -4.1006970, -2.0284209, -4.0794735, -2.0196781, -1.4969349, 1.4734530
9: -1.9327794, -0.2448264, -1.9223014, -0.2569355, -1.5330663, 1.5381274

Time for backsubstitution: 20.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382423, upper bound: 0.6368384
time: 3.64 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382425, upper bound: 0.6368384
time: 3.62 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.8298693, -6.6627150, -8.8258286, -6.7083549, -1.5598817, 1.5995893
1: -6.4107051, -4.6654797, -6.3870974, -4.6729341, -1.4315815, 1.4517851
2: -8.4337444, -6.8246889, -8.4221277, -6.8334188, -1.3630586, 1.3697016
3: -9.8073578, -7.9725561, -9.7897625, -7.9841328, -1.3020272, 1.3036067
4: -4.6977596, -2.9989305, -4.6715775, -3.0062649, -1.5303564, 1.5208378
5: -5.0719295, -3.3345690, -5.0665131, -3.3653073, -1.4095426, 1.4495590
6: -13.3927088, -11.1163502, -13.3296175, -11.1220188, -2.1009903, 2.0520725
7: 3.5806267, 4.8458314, 3.5952365, 4.8258848, -1.1628098, 1.1696432
8: -4.1097889, -2.0017138, -4.0810566, -2.0083556, -1.5190783, 1.4849792
9: -1.9433277, -0.2352464, -1.9260602, -0.2528281, -1.5487809, 1.5496776

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6406778, upper bound: 0.6359425
time: 3.61 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6406778, upper bound: 0.6359444
time: 3.69 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.8116322, -6.6657400, -8.8543339, -6.6986732, -1.5511584, 1.5880148
1: -6.4062147, -4.6702833, -6.3910341, -4.6721115, -1.4273005, 1.4496517
2: -8.4135923, -6.8309083, -8.4339590, -6.8268199, -1.3642883, 1.3710139
3: -9.7818203, -7.9840932, -9.8053236, -7.9770236, -1.2989192, 1.3064463
4: -4.6792669, -3.0052094, -4.6678772, -3.0042419, -1.5166578, 1.5140429
5: -5.0379100, -3.3481958, -5.0979824, -3.3552105, -1.4099731, 1.4240978
6: -13.3773823, -11.1210136, -13.3367872, -11.0896978, -2.0865288, 2.0338469
7: 3.6028566, 4.8430977, 3.5992393, 4.8366547, -1.1502259, 1.1614580
8: -4.1008110, -2.0274801, -4.1008453, -2.0117121, -1.5060706, 1.4954147
9: -1.9336275, -0.2447605, -1.9293826, -0.2441643, -1.5467610, 1.5505185

Time for backsubstitution: 23.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382590, upper bound: 0.6391616
time: 3.63 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6391616
time: 3.89 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.8299875, -6.6611123, -8.8620453, -6.6975536, -1.5690298, 1.6050048
1: -6.4108133, -4.6653619, -6.3921204, -4.6701903, -1.4342499, 1.4574475
2: -8.4339018, -6.8236637, -8.4427042, -6.8249083, -1.3807998, 1.3884063
3: -9.8074293, -7.9711194, -9.8164454, -7.9737339, -1.3174844, 1.3321242
4: -4.6982613, -2.9988289, -4.6760597, -3.0022609, -1.5345531, 1.5301113
5: -5.0721350, -3.3324919, -5.1129680, -3.3522429, -1.4211507, 1.4586406
6: -13.3942776, -11.1162863, -13.3434534, -11.0888643, -2.1046400, 2.0638390
7: 3.5799732, 4.8459311, 3.5897462, 4.8378191, -1.1753862, 1.1745801
8: -4.1098995, -2.0007706, -4.1024299, -2.0003810, -1.5282197, 1.5065939
9: -1.9441757, -0.2351789, -1.9331690, -0.2400632, -1.5624666, 1.5620475

Time for backsubstitution: 23.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6406945, upper bound: 0.6382575
time: 3.79 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6406945, upper bound: 0.6382590
time: 3.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.75 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6367101, upper bound: 0.6368384
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6367101, upper bound: 0.6383796
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6391441, upper bound: 0.6359444
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6391441, upper bound: 0.6397167
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6367267, upper bound: 0.6391634
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6367267, upper bound: 0.6406957
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6391621, upper bound: 0.6382590
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6391621, upper bound: 0.6420293
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6382423, upper bound: 0.6368384
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6382425, upper bound: 0.6368384
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6406778, upper bound: 0.6359425
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6406778, upper bound: 0.6359444
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6382590, upper bound: 0.6391616
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6391616
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6406945, upper bound: 0.6382575
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.75
Output dim: 7, lower bound: -0.6406945, upper bound: 0.6382590

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.7812729, -6.7066903, -8.7911434, -6.7110786, -1.5086117, 1.5232520
1: -6.3780766, -4.6916823, -6.3809776, -4.6893167, -1.3656449, 1.3672352
2: -8.3972359, -6.8407459, -8.4078903, -6.8406138, -1.3214226, 1.3305278
3: -9.7622290, -8.0013275, -9.7762222, -7.9999580, -1.2491164, 1.2604046
4: -4.6504025, -3.0343227, -4.6585360, -3.0303035, -1.4594898, 1.4608216
5: -5.0146055, -3.3740706, -5.0327621, -3.3720467, -1.3535914, 1.3680184
6: -13.3094540, -11.1653547, -13.3119535, -11.1618595, -1.9790726, 1.9739609
7: 3.6235583, 4.8215151, 3.6138542, 4.8226395, -1.1169000, 1.1257095
8: -4.0673990, -2.0506301, -4.0743256, -2.0404253, -1.4488282, 1.4487219
9: -1.9034801, -0.2649317, -1.9057177, -0.2597792, -1.5025516, 1.5004535

Time for backsubstitution: 23.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344074, upper bound: 0.6368374
time: 4.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344073, upper bound: 0.6368402
time: 8.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.7812729, -6.7066903, -8.8214970, -6.6717296, -1.5403647, 1.5541458
1: -6.3780766, -4.6916823, -6.4089699, -4.6681099, -1.3879218, 1.3950810
2: -8.3972359, -6.8407459, -8.4237137, -6.8316355, -1.3298349, 1.3462005
3: -9.7622290, -8.0013275, -9.7955093, -7.9828901, -1.2668941, 1.2804487
4: -4.6504025, -3.0343227, -4.6870804, -3.0013707, -1.4903779, 1.4893460
5: -5.0146055, -3.3740706, -5.0558119, -3.3477988, -1.3781815, 1.3914766
6: -13.3094540, -11.1653547, -13.3782616, -11.1176109, -2.0232253, 2.0385242
7: 3.6235583, 4.8215151, 3.5932434, 4.8441253, -1.1390319, 1.1468143
8: -4.0673990, -2.0506301, -4.1076488, -2.0176711, -1.4719558, 1.4814487
9: -1.9034801, -0.2649317, -1.9353971, -0.2397071, -1.5228529, 1.5303268

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344074, upper bound: 0.6383771
time: 4.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344073, upper bound: 0.6383796
time: 4.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.7994719, -6.7021661, -8.7954998, -6.7136669, -1.5291238, 1.5343995
1: -6.3826013, -4.6867723, -6.3803058, -4.6842690, -1.3845282, 1.3801327
2: -8.4175358, -6.8337283, -8.3994226, -6.8429871, -1.3406525, 1.3302393
3: -9.7877026, -7.9897089, -9.7631989, -8.0023470, -1.2726643, 1.2594099
4: -4.6691656, -3.0278020, -4.6504412, -3.0226643, -1.4829326, 1.4660640
5: -5.0487800, -3.3589232, -5.0239573, -3.3826194, -1.3819499, 1.3786855
6: -13.3263273, -11.1607065, -13.3078861, -11.1441603, -2.0116267, 1.9741755
7: 3.6010675, 4.8242927, 3.6220593, 4.8221741, -1.1398399, 1.1197498
8: -4.0764389, -2.0245070, -4.0697031, -2.0441842, -1.4565058, 1.4663677
9: -1.9136101, -0.2553413, -1.9081142, -0.2636707, -1.5092068, 1.5129888

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6368368, upper bound: 0.6359420
time: 4.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6368368, upper bound: 0.6359444
time: 3.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.7994719, -6.7021661, -8.8137608, -6.7091055, -1.5302801, 1.5504966
1: -6.3826013, -4.6867723, -6.3848524, -4.6793919, -1.3922968, 1.3870258
2: -8.4175358, -6.8337283, -8.4197044, -6.8358812, -1.3428016, 1.3461936
3: -9.7877026, -7.9897089, -9.7886868, -7.9901905, -1.2746685, 1.2742863
4: -4.6691656, -3.0278020, -4.6693087, -3.0161119, -1.4908142, 1.4839597
5: -5.0487800, -3.3589232, -5.0581164, -3.3672066, -1.3723922, 1.3877647
6: -13.3263273, -11.1607065, -13.3247433, -11.1394463, -2.0331211, 2.0084858
7: 3.6010675, 4.8242927, 3.5994949, 4.8249626, -1.1426859, 1.1434419
8: -4.0764389, -2.0245070, -4.0787849, -2.0178113, -1.4668450, 1.4618430
9: -1.9136101, -0.2553413, -1.9184648, -0.2540828, -1.5166669, 1.5213835

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6368371, upper bound: 0.6376177
time: 3.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6368371, upper bound: 0.6376176
time: 4.08 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.7813921, -6.7050924, -8.8273325, -6.7002568, -1.5177679, 1.5526633
1: -6.3781881, -4.6915593, -6.3860087, -4.6865387, -1.3684111, 1.3729255
2: -8.3973999, -6.8397183, -8.4284935, -6.8321671, -1.3390908, 1.3522542
3: -9.7622948, -7.9998951, -9.8029137, -7.9896040, -1.2646244, 1.2889299
4: -4.6509018, -3.0342195, -4.6629443, -3.0263493, -1.4636817, 1.4701385
5: -5.0148039, -3.3719912, -5.0791731, -3.3590093, -1.3651934, 1.3922193
6: -13.3110209, -11.1652918, -13.3259182, -11.1287670, -2.0136547, 1.9858947
7: 3.6229072, 4.8216128, 3.6083879, 4.8345766, -1.1294696, 1.1306262
8: -4.0675092, -2.0496888, -4.0956559, -2.0324421, -1.4651046, 1.4705050
9: -1.9043262, -0.2648637, -1.9127778, -0.2470089, -1.5162392, 1.5128598

Time for backsubstitution: 20.79 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.38 + 553.68 = 613.05 seconds
