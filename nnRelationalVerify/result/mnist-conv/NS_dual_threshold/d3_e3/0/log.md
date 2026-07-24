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
execution time: IAR + RelationalAnalysis = 22.06 + 37.57 = 59.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.6420365, upper bound: 0.6420360

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420327, upper bound: 0.6405168
time: 3.87 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420327, upper bound: 0.6420318
time: 4.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.96 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 8.96
Output dim: 7, lower bound: -0.6420327, upper bound: 0.6405168
NS_B2, status: Status.UNKNOWN, split count: 1, time: 8.96
Output dim: 7, lower bound: -0.6420327, upper bound: 0.6420318

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -8.8144798, -6.6996937, -8.7995949, -6.7005539, -1.5576982, 1.5445008
1: -6.3854980, -4.6786838, -6.3827133, -4.6866484, -1.3863111, 1.3914986
2: -8.4206667, -6.8298345, -8.4177094, -6.8326960, -1.3564882, 1.3564548
3: -9.7891169, -7.9817119, -9.7877798, -7.9882631, -1.2902117, 1.2959752
4: -4.6722927, -3.0155001, -4.6696796, -3.0276978, -1.4909616, 1.4973073
5: -5.0593328, -3.3549399, -5.0489936, -3.3568306, -1.4183512, 1.4118669
6: -13.3339672, -11.1390591, -13.3279057, -11.1606426, -2.0051618, 2.0224547
7: 3.5956483, 4.8255525, 3.6004035, 4.8243914, -1.1473682, 1.1437714
8: -4.0794468, -2.0122800, -4.0765510, -2.0235472, -1.4817667, 1.4905548
9: -1.9234422, -0.2536776, -1.9144611, -0.2552704, -1.5306220, 1.5228972

Time for backsubstitution: 21.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382606, upper bound: 0.6391629
time: 3.53 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420293, upper bound: 0.6405157
time: 4.35 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -8.8265467, -6.6989460, -8.8299942, -6.6611004, -1.6039009, 1.5740986
1: -6.3877411, -4.6722364, -6.4108148, -4.6653576, -1.4511690, 1.4332788
2: -8.4230814, -6.8273768, -8.4339132, -6.8236566, -1.3759613, 1.3767791
3: -9.7901926, -7.9756527, -9.8074360, -7.9711056, -1.3082736, 1.3233318
4: -4.6745548, -3.0056534, -4.6982703, -2.9988279, -1.5277772, 1.5368700
5: -5.0677176, -3.3530443, -5.0721436, -3.3324804, -1.4544401, 1.4490788
6: -13.3388329, -11.1216307, -13.3942900, -11.1162834, -2.0487404, 2.1047993
7: 3.5913973, 4.8264761, 3.5799646, 4.8459330, -1.1735656, 1.1639650
8: -4.0817299, -2.0028381, -4.1099014, -2.0007520, -1.5047121, 1.5331044
9: -1.9310335, -0.2524244, -1.9441811, -0.2351732, -1.5589108, 1.5522716

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 6208

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382606, upper bound: 0.6406948
time: 3.63 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420293, upper bound: 0.6420297
time: 3.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.40 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 29.40
Output dim: 7, lower bound: -0.6382606, upper bound: 0.6391629
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 29.40
Output dim: 7, lower bound: -0.6420293, upper bound: 0.6405157
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 29.40
Output dim: 7, lower bound: -0.6382606, upper bound: 0.6406948
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 29.40
Output dim: 7, lower bound: -0.6420293, upper bound: 0.6420297

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -8.7962112, -6.7042603, -8.7918587, -6.7016668, -1.5354323, 1.5298796
1: -6.3809566, -4.6835651, -6.3816290, -4.6885982, -1.3784032, 1.3852215
2: -8.4003887, -6.8369389, -8.4088545, -6.8345647, -1.3348832, 1.3406641
3: -9.7636261, -7.9938669, -9.7766447, -7.9914880, -1.2604735, 1.2713516
4: -4.6533794, -3.0220461, -4.6614995, -3.0296919, -1.4701676, 1.4785299
5: -5.0251403, -3.3703530, -5.0339632, -3.3597841, -1.3802319, 1.3791232
6: -13.3171024, -11.1437712, -13.3211861, -11.1614723, -1.9855824, 2.0050783
7: 3.6182427, 4.8227615, 3.6100147, 4.8232241, -1.1225624, 1.1305811
8: -4.0703716, -2.0386801, -4.0749807, -2.0349102, -1.4590406, 1.4636087
9: -1.9130929, -0.2632704, -1.9106959, -0.2593806, -1.5151320, 1.5085993

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382425, upper bound: 0.6368401
time: 3.91 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6391632
time: 4.19 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -8.8144741, -6.6996965, -8.7995949, -6.7005539, -1.5531492, 1.5444970
1: -6.3854980, -4.6786842, -6.3827143, -4.6866479, -1.3853426, 1.3930435
2: -8.4206600, -6.8298345, -8.4177065, -6.8326955, -1.3514986, 1.3554969
3: -9.7891102, -7.9817152, -9.7877779, -7.9882641, -1.2789509, 1.2959716
4: -4.6722865, -3.0155010, -4.6696796, -3.0276980, -1.4880514, 1.4945455
5: -5.0593252, -3.3549430, -5.0489912, -3.3568316, -1.3914289, 1.4106317
6: -13.3339624, -11.1390581, -13.3279057, -11.1606407, -2.0044150, 2.0351400
7: 3.5956554, 4.8255510, 3.6004045, 4.8243918, -1.1473629, 1.1438372
8: -4.0794449, -2.0122910, -4.0765510, -2.0235519, -1.4804254, 1.4746332
9: -1.9234395, -0.2536819, -1.9144609, -0.2552695, -1.5306177, 1.5201585

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405141, upper bound: 0.6405158
time: 4.08 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405141, upper bound: 0.6405143
time: 4.07 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -8.8082027, -6.7035732, -8.8222771, -6.6622171, -1.5812759, 1.5593734
1: -6.3831782, -4.6771002, -6.4097190, -4.6672912, -1.4431071, 1.4269743
2: -8.4027987, -6.8346090, -8.4251032, -6.8255367, -1.3541093, 1.3608947
3: -9.7646875, -7.9885230, -9.7962523, -7.9743843, -1.2784953, 1.2976706
4: -4.6554890, -3.0122213, -4.6901336, -3.0007341, -1.5067453, 1.5180240
5: -5.0335088, -3.3688040, -5.0571170, -3.3354266, -1.4162779, 1.4160273
6: -13.3219404, -11.1264257, -13.3876028, -11.1170874, -2.0291538, 2.0874090
7: 3.6141315, 4.8236709, 3.5893967, 4.8447547, -1.1486919, 1.1508632
8: -4.0726194, -2.0295391, -4.1083288, -2.0120821, -1.4818830, 1.5057249
9: -1.9203823, -0.2620188, -1.9404719, -0.2392797, -1.5430770, 1.5380406

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 6208

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382425, upper bound: 0.6383777
time: 3.67 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6406939
time: 3.72 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -8.8265409, -6.6989474, -8.8299932, -6.6611009, -1.5993297, 1.5740952
1: -6.3877420, -4.6722379, -6.4108148, -4.6653590, -1.4502034, 1.4347854
2: -8.4230728, -6.8273787, -8.4339104, -6.8236566, -1.3708868, 1.3758364
3: -9.7901859, -7.9756536, -9.8074360, -7.9711075, -1.2970119, 1.3233290
4: -4.6745501, -3.0056546, -4.6982703, -2.9988286, -1.5249505, 1.5341239
5: -5.0677109, -3.3530457, -5.0721407, -3.3324804, -1.4274392, 1.4478629
6: -13.3388271, -11.1216316, -13.3942890, -11.1162844, -2.0479937, 2.1120763
7: 3.5914030, 4.8264751, 3.5799661, 4.8459320, -1.1735606, 1.1640315
8: -4.0817289, -2.0028458, -4.1099014, -2.0007582, -1.5033746, 1.5168500
9: -1.9310309, -0.2524295, -1.9441824, -0.2351749, -1.5589070, 1.5495317

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6208

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420111, upper bound: 0.6397140
time: 3.58 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420278, upper bound: 0.6420275
time: 3.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.21 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6382425, upper bound: 0.6368401
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6391632
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6405141, upper bound: 0.6405158
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6405141, upper bound: 0.6405143
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6382425, upper bound: 0.6383777
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6406939
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6420111, upper bound: 0.6397140
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.21
Output dim: 7, lower bound: -0.6420278, upper bound: 0.6420275

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.7960920, -6.7058716, -8.7911434, -6.7110786, -1.5257564, 1.5272317
1: -6.3808455, -4.6836863, -6.3809776, -4.6893167, -1.3776565, 1.3844473
2: -8.4002256, -6.8379712, -8.4078903, -6.8406138, -1.3262768, 1.3353882
3: -9.7635536, -7.9953127, -9.7762222, -7.9999580, -1.2504210, 1.2666829
4: -4.6528783, -3.0221512, -4.6585360, -3.0303035, -1.4666238, 1.4743848
5: -5.0249395, -3.3724432, -5.0327621, -3.3720467, -1.3677068, 1.3754718
6: -13.3155308, -11.1438370, -13.3119535, -11.1618595, -1.9835701, 1.9957166
7: 3.6188951, 4.8226614, 3.6138542, 4.8226395, -1.1214237, 1.1265922
8: -4.0702600, -2.0396233, -4.0743256, -2.0404253, -1.4512405, 1.4596729
9: -1.9122435, -0.2633387, -1.9057177, -0.2597792, -1.5116420, 1.5020909

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367102, upper bound: 0.6368401
time: 3.60 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367102, upper bound: 0.6368401
time: 3.68 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.7962112, -6.7042699, -8.8273325, -6.7002568, -1.5349131, 1.5560277
1: -6.3809557, -4.6835670, -6.3860087, -4.6865387, -1.3804216, 1.3901379
2: -8.4003868, -6.8369446, -8.4284935, -6.8321671, -1.3439465, 1.3571153
3: -9.7636242, -7.9938784, -9.8029137, -7.9896040, -1.2659290, 1.2952085
4: -4.6533761, -3.0220473, -4.6629443, -3.0263493, -1.4708152, 1.4837027
5: -5.0251393, -3.3703632, -5.0791731, -3.3590093, -1.3793097, 1.3984377
6: -13.3170967, -11.1437702, -13.3259182, -11.1287670, -2.0181522, 2.0076499
7: 3.6182451, 4.8227601, 3.6083879, 4.8345766, -1.1339910, 1.1315098
8: -4.0703697, -2.0386848, -4.0956559, -2.0324421, -1.4675169, 1.4794950
9: -1.9130902, -0.2632706, -1.9127778, -0.2470089, -1.5253301, 1.5144970

Time for backsubstitution: 21.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367269, upper bound: 0.6391633
time: 4.37 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6367269, upper bound: 0.6391613
time: 4.45 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -8.7995911, -6.7005553, -8.7995949, -6.7005539, -1.5358853, 1.5404329
1: -6.3827119, -4.6866493, -6.3827143, -4.6866479, -1.3733010, 1.3757792
2: -8.4177008, -6.8326955, -8.4177065, -6.8326955, -1.3464832, 1.3505049
3: -9.7877741, -7.9882655, -9.7877779, -7.9882641, -1.2776358, 1.2888925
4: -4.6696739, -3.0276990, -4.6696796, -3.0276980, -1.4807696, 1.4808831
5: -5.0489864, -3.3568318, -5.0489912, -3.3568316, -1.3772612, 1.4029412
6: -13.3279009, -11.1606407, -13.3279057, -11.1606407, -1.9998956, 2.0133262
7: 3.6004105, 4.8243914, 3.6004045, 4.8243918, -1.1428688, 1.1429396
8: -4.0765495, -2.0235586, -4.0765510, -2.0235519, -1.4779539, 1.4633279
9: -1.9144585, -0.2552743, -1.9144609, -0.2552695, -1.5212541, 1.5185208

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404959, upper bound: 0.6381981
time: 3.76 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405126, upper bound: 0.6405140
time: 3.81 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -8.8299236, -6.6612043, -8.7995949, -6.7005539, -1.5667577, 1.5710905
1: -6.4107103, -4.6654758, -6.3827143, -4.6866479, -1.4011765, 1.3979800
2: -8.4334621, -6.8237205, -8.4177065, -6.8326955, -1.3621030, 1.3589911
3: -9.8071146, -7.9711308, -9.7877779, -7.9882641, -1.2977145, 1.3067091
4: -4.6981554, -2.9988699, -4.6696796, -3.0276980, -1.5092745, 1.5116568
5: -5.0720329, -3.3325973, -5.0489912, -3.3568316, -1.4007149, 1.4275527
6: -13.3941975, -11.1164312, -13.3279057, -11.1606407, -2.0635819, 2.0574460
7: 3.5799870, 4.8458939, 3.6004045, 4.8243918, -1.1638525, 1.1650872
8: -4.1098909, -2.0008373, -4.0765510, -2.0235519, -1.5062442, 1.4867024
9: -1.9440913, -0.2352026, -1.9144609, -0.2552695, -1.5510478, 1.5388167

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404959, upper bound: 0.6381982
time: 4.14 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405126, upper bound: 0.6405123
time: 3.76 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.8080807, -6.7051849, -8.8215618, -6.6716270, -1.5715911, 1.5567217
1: -6.3830681, -4.6772180, -6.4090729, -4.6679945, -1.4423451, 1.4262016
2: -8.4026365, -6.8356404, -8.4241543, -6.8315740, -1.3454914, 1.3556070
3: -9.7646160, -7.9899678, -9.7958241, -7.9828672, -1.2684312, 1.2929993
4: -4.6549864, -3.0123255, -4.6871896, -3.0013301, -1.5031729, 1.5138965
5: -5.0333095, -3.3708949, -5.0559154, -3.3476825, -1.4037628, 1.4123685
6: -13.3203688, -11.1264935, -13.3783484, -11.1174650, -2.0271497, 2.0781021
7: 3.6147833, 4.8235703, 3.5932269, 4.8441625, -1.1475463, 1.1468897
8: -4.0725050, -2.0304828, -4.1076565, -2.0175967, -1.4740787, 1.5015564
9: -1.9195323, -0.2620854, -1.9354843, -0.2396840, -1.5395880, 1.5315442

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 6208

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6359428, upper bound: 0.6383766
time: 3.69 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6359428, upper bound: 0.6383767
time: 3.83 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.8082018, -6.7035809, -8.8577995, -6.6608438, -1.5807674, 1.5795598
1: -6.3831782, -4.6771002, -6.4141874, -4.6651487, -1.4450130, 1.4319906
2: -8.4027977, -6.8346138, -8.4447870, -6.8229952, -1.3631673, 1.3742659
3: -9.7646875, -7.9885325, -9.8224697, -7.9725270, -1.2839122, 1.3118838
4: -4.6554856, -3.0122223, -4.6916304, -2.9973261, -1.5073209, 1.5228074
5: -5.0335088, -3.3688147, -5.1024070, -3.3346343, -1.4153757, 1.4248922
6: -13.3219328, -11.1264267, -13.3921366, -11.0842991, -2.0618114, 2.0901875
7: 3.6141348, 4.8236699, 3.5877388, 4.8560581, -1.1600814, 1.1518362
8: -4.0726185, -2.0295434, -4.1289258, -2.0095901, -1.4903908, 1.5054178
9: -1.9203794, -0.2620189, -1.9426718, -0.2269347, -1.5532660, 1.5439334

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 6208

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6382572
time: 3.72 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6406939
time: 3.72 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.8264198, -6.7005568, -8.8292780, -6.6705146, -1.5896387, 1.5714450
1: -6.3876305, -4.6723576, -6.4101706, -4.6660647, -1.4494362, 1.4340072
2: -8.4229126, -6.8284087, -8.4329634, -6.8296943, -1.3622251, 1.3705425
3: -9.7901125, -7.9770989, -9.8070059, -7.9795909, -1.2869556, 1.3186615
4: -4.6740417, -3.0057578, -4.6953096, -2.9994218, -1.5213757, 1.5300069
5: -5.0675054, -3.3551347, -5.0709248, -3.3447342, -1.4149165, 1.4441996
6: -13.3372583, -11.1216974, -13.3850336, -11.1166630, -2.0459886, 2.1027694
7: 3.5920591, 4.8263760, 3.5838010, 4.8453407, -1.1724095, 1.1600540
8: -4.0816135, -2.0037932, -4.1092343, -2.0062799, -1.4955711, 1.5126746
9: -1.9301805, -0.2524962, -1.9391961, -0.2355770, -1.5554194, 1.5430450

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6208

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6397136
time: 3.95 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6397144
time: 4.15 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.8265419, -6.6989541, -8.8655119, -6.6597261, -1.5988259, 1.5965724
1: -6.3877411, -4.6722383, -6.4152846, -4.6632152, -1.4521008, 1.4397786
2: -8.4230709, -6.8273840, -8.4535837, -6.8210912, -1.3798885, 1.3915529
3: -9.7901850, -7.9756651, -9.8336401, -7.9692259, -1.3024209, 1.3400817
4: -4.6745467, -3.0056548, -4.6997852, -2.9954219, -1.5254989, 1.5393271
5: -5.0677090, -3.3530560, -5.1173959, -3.3316779, -1.4265313, 1.4594419
6: -13.3388186, -11.1216316, -13.3987904, -11.0834970, -2.0806484, 2.1148267
7: 3.5914059, 4.8264751, 3.5783045, 4.8572330, -1.1849480, 1.1650074
8: -4.0817294, -2.0028524, -4.1305032, -1.9982643, -1.5118809, 1.5165484
9: -1.9310273, -0.2524292, -1.9464136, -0.2228351, -1.5690880, 1.5554116

Time for backsubstitution: 22.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6420111
time: 3.83 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6420286
time: 4.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.63 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6367102, upper bound: 0.6368401
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6367102, upper bound: 0.6368401
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6367269, upper bound: 0.6391633
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6367269, upper bound: 0.6391613
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6404959, upper bound: 0.6381981
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6405126, upper bound: 0.6405140
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6404959, upper bound: 0.6381982
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6405126, upper bound: 0.6405123
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6359428, upper bound: 0.6383766
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6359428, upper bound: 0.6383767
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6382572
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6382592, upper bound: 0.6406939
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6397136
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6397144
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6420111
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.63
Output dim: 7, lower bound: -0.6397136, upper bound: 0.6420286

## BFS NS instance: NS_B1_A1_B1_A1

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

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6368377
time: 4.16 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6368383
time: 4.59 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.8113613, -6.6674805, -8.7911434, -6.7110786, -1.5390100, 1.5509861
1: -6.4059849, -4.6705284, -6.3809776, -4.6893167, -1.3934121, 1.3895063
2: -8.4129305, -6.8320794, -8.4078903, -6.8406138, -1.3368182, 1.3384767
3: -9.7814159, -7.9860940, -9.7762222, -7.9999580, -1.2690768, 1.2755299
4: -4.6785192, -3.0053720, -4.6585360, -3.0303035, -1.4876070, 1.4916120
5: -5.0375810, -3.3506329, -5.0327621, -3.3720467, -1.3768616, 1.3890657
6: -13.3756809, -11.1212921, -13.3119535, -11.1618595, -2.0431104, 2.0179448
7: 3.6038411, 4.8429456, 3.6138542, 4.8226395, -1.1372838, 1.1477919
8: -4.1006622, -2.0287995, -4.0743256, -2.0404253, -1.4760785, 1.4708786
9: -1.9324515, -0.2448542, -1.9057177, -0.2597792, -1.5315580, 1.5207567

Time for backsubstitution: 21.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B1_A1_B1_A2_A1

### Relational analysis result of NS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6368378
time: 4.23 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2

### Relational analysis result of NS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6368401
time: 3.72 seconds

## BFS NS instance: NS_B1_A1_B2_A1

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

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6391453
time: 4.21 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6368387
time: 4.24 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8114815, -6.6658797, -8.8273325, -6.7002568, -1.5481668, 1.5564308
1: -6.4060955, -4.6704092, -6.3860087, -4.6865387, -1.3961768, 1.3951936
2: -8.4130907, -6.8310533, -8.4284935, -6.8321671, -1.3544855, 1.3599887
3: -9.7814875, -7.9846592, -9.8029137, -7.9896040, -1.2845857, 1.3027761
4: -4.6790152, -3.0052712, -4.6629443, -3.0263493, -1.4917965, 1.5009270
5: -5.0377808, -3.3485556, -5.0791731, -3.3590093, -1.3884621, 1.3981535
6: -13.3772507, -11.1212273, -13.3259182, -11.1287670, -2.0467339, 2.0298772
7: 3.6031911, 4.8430452, 3.6083879, 4.8345766, -1.1498525, 1.1527095
8: -4.1007757, -2.0278616, -4.0956559, -2.0324421, -1.4852295, 1.4834960
9: -1.9332986, -0.2447857, -1.9127778, -0.2470089, -1.5452456, 1.5331612

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of NS_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6391435
time: 4.19 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6344075, upper bound: 0.6369347
time: 3.74 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.63 + 541.66 = 601.30 seconds
