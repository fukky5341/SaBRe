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
execution time: IAR + RelationalAnalysis = 22.08 + 38.12 = 60.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.6420365, upper bound: 0.6420360

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6414674, upper bound: 0.6420349
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420356, upper bound: 0.6414689
time: 3.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.56 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.56
Output dim: 7, lower bound: -0.6414674, upper bound: 0.6420349
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.56
Output dim: 7, lower bound: -0.6420356, upper bound: 0.6414689

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4933233, 1.5076404
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4308934, 1.4282594
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3705006, 1.3703885
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2960663, 1.2942343
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4885559, 1.4810662
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4199800, 1.4266186
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0554948, 2.0576544
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1538253, 1.1536746
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5050192, 1.5051260
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5353928, 1.5334752

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6414668, upper bound: 0.6417591
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6411897, upper bound: 0.6420348
time: 4.17 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5076404, 1.4933233
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4282594, 1.4308934
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3703885, 1.3705006
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2942343, 1.2960660
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4810667, 1.4885554
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4266186, 1.4199800
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0576539, 2.0554948
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1536746, 1.1538253
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5051260, 1.5050192
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5334754, 1.5353928

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6398011, upper bound: 0.6414654
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420341, upper bound: 0.6392319
time: 3.34 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.08 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.08
Output dim: 7, lower bound: -0.6414668, upper bound: 0.6417591
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.08
Output dim: 7, lower bound: -0.6411897, upper bound: 0.6420348
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.08
Output dim: 7, lower bound: -0.6398011, upper bound: 0.6414654
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.08
Output dim: 7, lower bound: -0.6420341, upper bound: 0.6392319

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4947391, 1.5095482
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4313326, 1.4288464
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3701649, 1.3701088
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2961602, 1.2943051
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4864826, 1.4785728
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4187570, 1.4255972
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0550499, 2.0572848
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1537457, 1.1535578
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5041075, 1.5040350
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5356641, 1.5336771

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399475, upper bound: 0.6417557
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6414629, upper bound: 0.6402394
time: 3.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4952326, 1.5090561
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4314804, 1.4286988
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3702211, 1.3700531
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2961369, 1.2943285
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4860625, 1.4789944
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4189587, 1.4253955
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0551252, 2.0572095
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1537085, 1.1535950
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5039282, 1.5042143
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5355945, 1.5337462

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 874

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6342778, upper bound: 0.6419489
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6411044, upper bound: 0.6351227
time: 6.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5059047, 1.4850130
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4282560, 1.4308772
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3688745, 1.3632202
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2924776, 1.2876105
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4786396, 1.4880590
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4248672, 1.4115458
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0499759, 2.0539050
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1514730, 1.1533678
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5036988, 1.4981632
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5286307, 1.5343881

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 551

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6398009, upper bound: 0.6411887
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6395273, upper bound: 0.6414654
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4993296, 1.4915881
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4282432, 1.4308898
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3631086, 1.3689866
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2857785, 1.2943096
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4805698, 1.4861288
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4181848, 1.4182281
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0560651, 2.0478163
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1532168, 1.1516237
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4982700, 1.5035920
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5324702, 1.5305481

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 874

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6351221, upper bound: 0.6391467
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6419488, upper bound: 0.6323196
time: 3.73 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.63 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6399475, upper bound: 0.6417557
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6414629, upper bound: 0.6402394
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6342778, upper bound: 0.6419489
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6411044, upper bound: 0.6351227
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6398009, upper bound: 0.6411887
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6395273, upper bound: 0.6414654
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6351221, upper bound: 0.6391467
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.63
Output dim: 7, lower bound: -0.6419488, upper bound: 0.6323196

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4671202, 1.4938941
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4161229, 1.4019575
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3667712, 1.3641179
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2953191, 1.2928231
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4746237, 1.4576273
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.3975048, 1.4135733
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0507388, 2.0496840
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1509986, 1.1519992
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5016322, 1.4996772
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5324707, 1.5318666

Time for backsubstitution: 22.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399460, upper bound: 0.6417542
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399460, upper bound: 0.6417518
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4791012, 1.4819288
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4044437, 1.4136395
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3641744, 1.3667142
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2946787, 1.2934644
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4655361, 1.4667220
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4067354, 1.4043450
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0474486, 2.0529757
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1521873, 1.1508107
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4997497, 1.5015588
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5338540, 1.5304837

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6414615, upper bound: 0.6402379
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6414615, upper bound: 0.6402359
time: 3.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4965129, 1.5087900
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4461155, 1.4411972
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3682947, 1.3687356
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3036890, 1.3037326
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4944515, 1.4855514
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4190245, 1.4253829
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0536423, 2.0559382
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1451445, 1.1464553
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4980698, 1.4971876
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5271668, 1.5234711

Time for backsubstitution: 22.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 551

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6342774, upper bound: 0.6416702
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6342752, upper bound: 0.6419493
time: 4.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4949670, 1.5103364
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4439788, 1.4433336
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3689032, 1.3681271
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3055410, 1.3018804
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4926200, 1.4873834
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4189463, 1.4254611
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0538540, 2.0557265
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1465688, 1.1450307
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4969015, 1.4983559
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5253196, 1.5253181

Time for backsubstitution: 22.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6411029, upper bound: 0.6351231
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6411029, upper bound: 0.6351213
time: 4.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4799862, 1.4539208
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4290915, 1.4310389
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3593612, 1.3554347
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2395122, 1.2434771
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4773321, 1.4864912
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4139085, 1.4037318
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0491071, 2.0532646
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1524291, 1.1540821
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4640269, 1.4505620
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5193377, 1.5266838

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397996, upper bound: 0.6411868
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397996, upper bound: 0.6411871
time: 3.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4748125, 1.4590979
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4284177, 1.4317126
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3610892, 1.3537071
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2483425, 1.2346454
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4770718, 1.4867516
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4170537, 1.4005876
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0493360, 2.0530362
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1521869, 1.1543238
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4560981, 1.4584956
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5209284, 1.5250950

Time for backsubstitution: 22.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 874

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6326152, upper bound: 0.6413796
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6394420, upper bound: 0.6345534
time: 3.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5006108, 1.4913225
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4428787, 1.4433887
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3611836, 1.3676705
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2933302, 1.3037136
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4889588, 1.4926858
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4182501, 1.4182153
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0545821, 2.0465446
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1446526, 1.1444840
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4924102, 1.4965639
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5240421, 1.5202730

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6351208, upper bound: 0.6391456
time: 5.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6351208, upper bound: 0.6391456
time: 5.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4990644, 1.4928689
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4407420, 1.4455252
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3617926, 1.3670616
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2951827, 1.3018613
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4871273, 1.4945178
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4181719, 1.4182935
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0547938, 2.0463328
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1460774, 1.1430593
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4912419, 1.4977326
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5221953, 1.5221200

Time for backsubstitution: 22.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6404297, upper bound: 0.6323156
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6419449, upper bound: 0.6307996
time: 3.60 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6399460, upper bound: 0.6417542
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6399460, upper bound: 0.6417518
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6414615, upper bound: 0.6402379
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6414615, upper bound: 0.6402359
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6342774, upper bound: 0.6416702
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6342752, upper bound: 0.6419493
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6411029, upper bound: 0.6351231
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6411029, upper bound: 0.6351213
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6397996, upper bound: 0.6411868
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6397996, upper bound: 0.6411871
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6326152, upper bound: 0.6413796
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6394420, upper bound: 0.6345534
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6351208, upper bound: 0.6391456
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6351208, upper bound: 0.6391456
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6404297, upper bound: 0.6323156
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 7, lower bound: -0.6419449, upper bound: 0.6307996

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4631329, 1.4891229
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4123111, 1.3973844
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3671460, 1.3650661
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2949038, 1.2924769
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4744658, 1.4575863
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.3970776, 1.4126229
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0222125, 2.0259137
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1514173, 1.1526804
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4981890, 1.4929414
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5322089, 1.5317960

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 551

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399457, upper bound: 0.6417511
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6396676, upper bound: 0.6417518
time: 7.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4623485, 1.4898973
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4115496, 1.3981428
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3677197, 1.3644924
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2949734, 1.2924080
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4745741, 1.4574699
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.3965545, 1.4131417
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0269685, 2.0211573
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1516786, 1.1524179
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4948959, 1.4962306
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5323987, 1.5316050

Time for backsubstitution: 20.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6378452, upper bound: 0.6417495
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399428, upper bound: 0.6396516
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4751139, 1.4771576
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4006324, 1.4090662
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3645492, 1.3676629
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2942638, 1.2931182
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4653792, 1.4666815
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4063082, 1.4033947
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0189214, 2.0292053
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1526055, 1.1514916
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4963069, 1.4948230
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5335927, 1.5304132

Time for backsubstitution: 21.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 874

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6345496, upper bound: 0.6401508
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6413762, upper bound: 0.6333237
time: 4.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.4743295, 1.4779320
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.3998709, 1.4098248
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3651228, 1.3670893
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2943325, 1.2930491
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4654870, 1.4665647
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4057851, 1.4039135
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0236783, 2.0244484
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1528673, 1.1512294
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4930139, 1.4981122
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5337825, 1.5302222

Time for backsubstitution: 21.52 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 60.20 + 543.22 = 603.42 seconds
