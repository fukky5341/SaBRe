## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.41043252599999996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8283465, 0.8283467)
1: (-7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7843864, 0.7843866)
2: (-4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7637737, 0.7637737)
3: (-6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9626312, 0.9626315)
4: (-12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9123442, 0.9123440)
5: (-6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5031065, 0.5031066)
6: (-5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7319636, 0.7319636)
7: (-11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8130295, 0.8130298)
8: (9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6571708, 0.6571705)
9: (-7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9103527, 0.9103529)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.74 + 32.88 = 55.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4188075, upper bound: 0.4188087

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4611
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4611

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174989, upper bound: 0.4188038
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188026, upper bound: 0.4175002
time: 3.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.34 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.34
Output dim: 8, lower bound: -0.4174989, upper bound: 0.4188038
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.34
Output dim: 8, lower bound: -0.4188026, upper bound: 0.4175002

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8247635, 0.8268802
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7824545, 0.7796460
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7636349, 0.7634428
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9607310, 0.9579659
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9082365, 0.9106755
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5007246, 0.4972520
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7294142, 0.7257082
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8111985, 0.8122828
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6563678, 0.6568425
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9085588, 0.9059355

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5843

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4169414, upper bound: 0.4187996
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174959, upper bound: 0.4182444
time: 3.60 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8268805, 0.8247635
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7796462, 0.7824545
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7634432, 0.7636344
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9579659, 0.9607310
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9106755, 0.9082367
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4972522, 0.5007246
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7257082, 0.7294145
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8122828, 0.8111985
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6568422, 0.6563678
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9059353, 0.9085586

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5843

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182432, upper bound: 0.4174972
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4187995, upper bound: 0.4169427
time: 3.12 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.78
Output dim: 8, lower bound: -0.4169414, upper bound: 0.4187996
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.78
Output dim: 8, lower bound: -0.4174959, upper bound: 0.4182444
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.78
Output dim: 8, lower bound: -0.4182432, upper bound: 0.4174972
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.78
Output dim: 8, lower bound: -0.4187995, upper bound: 0.4169427

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8234613, 0.8306873
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7840900, 0.7790897
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7640059, 0.7633138
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9658027, 0.9562356
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9080927, 0.9110975
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5021536, 0.4967630
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7281294, 0.7294674
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8116779, 0.8121166
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6557693, 0.6585953
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9075432, 0.9089022

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4158525, upper bound: 0.4187989
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4169395, upper bound: 0.4177093
time: 5.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8247635, 0.8255780
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7818983, 0.7796460
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7635057, 0.7634428
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9590006, 0.9579659
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9082365, 0.9105315
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5002353, 0.4972520
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7294142, 0.7244234
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8110323, 0.8122828
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6563678, 0.6562443
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9085588, 0.9049206

Time for backsubstitution: 22.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4164070, upper bound: 0.4182411
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174940, upper bound: 0.4171535
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8255780, 0.8285704
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7812817, 0.7818981
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7638142, 0.7635055
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9630375, 0.9590006
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9105318, 0.9086587
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4986808, 0.5002356
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7244234, 0.7331736
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8127623, 0.8110323
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6562443, 0.6581206
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9049206, 0.9115252

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4171522, upper bound: 0.4174953
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182413, upper bound: 0.4164083
time: 3.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8268805, 0.8234613
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7790897, 0.7824545
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7633140, 0.7636344
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9562359, 0.9607310
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9106755, 0.9080930
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4967630, 0.5007246
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7257082, 0.7281294
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8121166, 0.8111985
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6568422, 0.6557693
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9059353, 0.9075432

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177092, upper bound: 0.4169397
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4187976, upper bound: 0.4158538
time: 3.39 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4158525, upper bound: 0.4187989
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4169395, upper bound: 0.4177093
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4164070, upper bound: 0.4182411
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4174940, upper bound: 0.4171535
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4171522, upper bound: 0.4174953
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4182413, upper bound: 0.4164083
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4177092, upper bound: 0.4169397
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.06
Output dim: 8, lower bound: -0.4187976, upper bound: 0.4158538

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8219731, 0.8299854
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7840629, 0.7790233
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7630112, 0.7628415
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9657969, 0.9562330
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9076786, 0.9102118
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5010962, 0.4962647
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7277167, 0.7286098
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8106742, 0.8116264
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6539640, 0.6577454
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9072647, 0.9087691

Time for backsubstitution: 21.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4158459, upper bound: 0.4180083
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4150606, upper bound: 0.4187926
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8227592, 0.8291991
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7840238, 0.7790625
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7635338, 0.7623193
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9657998, 0.9562302
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9072070, 0.9106832
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5016551, 0.4957056
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7272718, 0.7290547
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8111877, 0.8111126
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6549196, 0.6567898
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9074101, 0.9086237

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4169330, upper bound: 0.4169210
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4161476, upper bound: 0.4177042
time: 4.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8232753, 0.8248761
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7818708, 0.7795801
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7625105, 0.7629704
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9589953, 0.9579635
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9078224, 0.9096458
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4991782, 0.4967537
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7290018, 0.7235656
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8100281, 0.8117929
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6545625, 0.6553946
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9082794, 0.9047875

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4164006, upper bound: 0.4174526
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4156153, upper bound: 0.4182361
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8240614, 0.8240902
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7818317, 0.7796193
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7630332, 0.7624481
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9589982, 0.9579606
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9073513, 0.9101171
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4997373, 0.4961947
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7285566, 0.7240107
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8105421, 0.8112788
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6555181, 0.6544387
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9084253, 0.9046419

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174878, upper bound: 0.4163639
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4167024, upper bound: 0.4171469
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8240900, 0.8278682
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7812545, 0.7818317
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7628195, 0.7630332
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9630322, 0.9589982
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9101171, 0.9077730
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4976237, 0.4997371
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7240107, 0.7323158
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8117580, 0.8105421
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6544385, 0.6572707
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9046421, 0.9113920

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4171457, upper bound: 0.4167036
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163627, upper bound: 0.4174890
time: 3.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8248761, 0.8270824
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7812152, 0.7818708
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7633421, 0.7625110
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9630351, 0.9589953
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9096456, 0.9082444
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4981828, 0.4991782
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7235656, 0.7327609
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8122721, 0.8100283
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6553946, 0.6563148
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9047875, 0.9112465

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182349, upper bound: 0.4156166
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174514, upper bound: 0.4164019
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8253920, 0.8227594
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7790625, 0.7823887
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7623188, 0.7631621
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9562306, 0.9607284
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9102609, 0.9072070
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4957056, 0.5002263
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7252955, 0.7272718
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8111124, 0.8107085
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6550364, 0.6549196
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9056568, 0.9074101

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177030, upper bound: 0.4161489
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4169198, upper bound: 0.4169343
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8261783, 0.8219733
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7790234, 0.7824278
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7628415, 0.7626398
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9562335, 0.9607255
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9097898, 0.9076786
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4962647, 0.4996673
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7248504, 0.7277169
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8116264, 0.8101945
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6559920, 0.6539640
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9058018, 0.9072647

Time for backsubstitution: 22.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4187914, upper bound: 0.4150619
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4180072, upper bound: 0.4158471
time: 3.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4158459, upper bound: 0.4180083
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4150606, upper bound: 0.4187926
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4169330, upper bound: 0.4169210
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4161476, upper bound: 0.4177042
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4164006, upper bound: 0.4174526
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4156153, upper bound: 0.4182361
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4174878, upper bound: 0.4163639
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4167024, upper bound: 0.4171469
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4171457, upper bound: 0.4167036
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4163627, upper bound: 0.4174890
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4182349, upper bound: 0.4156166
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4174514, upper bound: 0.4164019
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4177030, upper bound: 0.4161489
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4169198, upper bound: 0.4169343
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4187914, upper bound: 0.4150619
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 8, lower bound: -0.4180072, upper bound: 0.4158471

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8219664, 0.8299768
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7840631, 0.7790239
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7630105, 0.7628429
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9657984, 0.9562349
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9076595, 0.9101965
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5010962, 0.4962646
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7277229, 0.7286141
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8106723, 0.8116264
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6539652, 0.6577463
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9072728, 0.9087801

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1251
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 628

Time for candidate selection: 0.50 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4053864, upper bound: 0.4075335
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4053864, upper bound: 0.4075335
time: 3.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8219650, 0.8299785
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7840633, 0.7790235
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7630124, 0.7628410
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9657989, 0.9562347
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9076633, 0.9101930
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5010962, 0.4962645
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7277212, 0.7286158
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8106742, 0.8116248
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6539648, 0.6577466
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9072757, 0.9087772

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1251
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 628

Time for candidate selection: 0.46 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4045994, upper bound: 0.4083195
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4045994, upper bound: 0.4083195
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8227525, 0.8291910
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7840240, 0.7790630
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7635331, 0.7623208
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9658012, 0.9562321
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9071884, 0.9106681
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5016551, 0.4957056
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7272778, 0.7290592
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8111863, 0.8111126
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6549208, 0.6567905
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9074183, 0.9086347

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1251
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 628

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4064647, upper bound: 0.4064535
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4064647, upper bound: 0.4064532
time: 4.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8227508, 0.8291924
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7840242, 0.7790626
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7635350, 0.7623186
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9658017, 0.9562318
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9071918, 0.9106643
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5016551, 0.4957056
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7272761, 0.7290609
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8111877, 0.8111107
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6549203, 0.6567907
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9074211, 0.9086318

Time for backsubstitution: 22.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1251
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 628

Time for candidate selection: 0.46 seconds

### Candidate
type: DSZ, layer: 3, pos: 1403

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4056787, upper bound: 0.4072402
time: 5.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4056787, upper bound: 0.4072414
time: 3.45 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 31.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4053864, upper bound: 0.4075335
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4053864, upper bound: 0.4075335
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4045994, upper bound: 0.4083195
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4045994, upper bound: 0.4083195
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4064647, upper bound: 0.4064535
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4064647, upper bound: 0.4064532
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4056787, upper bound: 0.4072402
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.62
Output dim: 8, lower bound: -0.4056787, upper bound: 0.4072414
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4164006, upper bound: 0.4174526
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4156153, upper bound: 0.4182361
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4174878, upper bound: 0.4163639
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4167024, upper bound: 0.4171469
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4171457, upper bound: 0.4167036
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4163627, upper bound: 0.4174890
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4182349, upper bound: 0.4156166
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4174514, upper bound: 0.4164019
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4177030, upper bound: 0.4161489
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4169198, upper bound: 0.4169343
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4187914, upper bound: 0.4150619
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 8, lower bound: -0.4180072, upper bound: 0.4158471

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.62 + 546.17 = 601.79 seconds
