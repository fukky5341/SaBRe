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
execution time: IAR + RelationalAnalysis = 24.55 + 32.89 = 57.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4188075, upper bound: 0.4188087

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5843
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 4611
type: DSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5843

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182481, upper bound: 0.4188056
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188046, upper bound: 0.4182493
time: 3.85 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.46
Output dim: 8, lower bound: -0.4182481, upper bound: 0.4188056
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.46
Output dim: 8, lower bound: -0.4188046, upper bound: 0.4182493

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8270450, 0.8321540
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7860212, 0.7838295
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7641451, 0.7636449
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9677038, 0.9609017
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9121997, 0.9127660
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5045356, 0.5026175
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7306788, 0.7357228
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8135090, 0.8128633
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6565728, 0.6589241
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9093380, 0.9133196

Time for backsubstitution: 22.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 4611

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182417, upper bound: 0.4180152
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174582, upper bound: 0.4187994
time: 2.99 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8283465, 0.8270450
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7838295, 0.7843866
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7636445, 0.7637737
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9609017, 0.9626315
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9123442, 0.9122000
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5026172, 0.5031066
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7319636, 0.7306788
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8128633, 0.8130298
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6571708, 0.6565731
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9103527, 0.9093380

Time for backsubstitution: 22.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4611
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4611

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174959, upper bound: 0.4182444
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4187995, upper bound: 0.4169427
time: 3.01 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.25 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.25
Output dim: 8, lower bound: -0.4182417, upper bound: 0.4180152
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.25
Output dim: 8, lower bound: -0.4174582, upper bound: 0.4187994
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.25
Output dim: 8, lower bound: -0.4174959, upper bound: 0.4182444
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.25
Output dim: 8, lower bound: -0.4187995, upper bound: 0.4169427

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8270385, 0.8321462
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7860215, 0.7838297
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7641449, 0.7636466
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9677038, 0.9609022
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9121811, 0.9127507
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5045354, 0.5026172
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7306845, 0.7357271
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8135071, 0.8128631
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6565735, 0.6589241
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9093466, 0.9133313

Time for backsubstitution: 22.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 4611

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4171507, upper bound: 0.4180133
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182398, upper bound: 0.4169260
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8270369, 0.8321476
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7860215, 0.7838295
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7641473, 0.7636445
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9677043, 0.9609020
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9121850, 0.9127471
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5045354, 0.5026172
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7306828, 0.7357287
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8135090, 0.8128614
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6565731, 0.6589243
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9093494, 0.9133282

Time for backsubstitution: 22.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 4611

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163676, upper bound: 0.4187975
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174564, upper bound: 0.4177091
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 22.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174896, upper bound: 0.4174545
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4167042, upper bound: 0.4182367
time: 5.56 seconds

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

Time for backsubstitution: 22.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177092, upper bound: 0.4169397
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4187976, upper bound: 0.4158538
time: 3.26 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4171507, upper bound: 0.4180133
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4182398, upper bound: 0.4169260
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4163676, upper bound: 0.4187975
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4174564, upper bound: 0.4177091
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4174896, upper bound: 0.4174545
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4167042, upper bound: 0.4182367
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4177092, upper bound: 0.4169397
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 8, lower bound: -0.4187976, upper bound: 0.4158538

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8255503, 0.8314438
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7859952, 0.7837644
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7631507, 0.7631743
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9676986, 0.9608998
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9117670, 0.9118652
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5034778, 0.5021189
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7302723, 0.7348697
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8125036, 0.8123732
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6547682, 0.6580746
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9090672, 0.9131978

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4611

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4611

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4158459, upper bound: 0.4180083
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4171457, upper bound: 0.4167036
time: 3.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8263364, 0.8306580
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7859559, 0.7838035
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7636724, 0.7626519
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9677014, 0.9608970
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9112954, 0.9123363
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5040369, 0.5015600
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7298272, 0.7353148
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8130176, 0.8118594
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6557238, 0.6571188
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9092131, 0.9130523

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4611

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4611

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4169330, upper bound: 0.4169210
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182349, upper bound: 0.4156166
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8255489, 0.8314455
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7859952, 0.7837640
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7631526, 0.7631721
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9676991, 0.9608996
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9117708, 0.9118614
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5034781, 0.5021188
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7302709, 0.7348714
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8125050, 0.8123717
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6547678, 0.6580749
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9090710, 0.9131949

Time for backsubstitution: 22.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4611

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4611

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4150606, upper bound: 0.4187926
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163627, upper bound: 0.4174890
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8263347, 0.8306594
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7859561, 0.7838032
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7636743, 0.7626498
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9677019, 0.9608967
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9112992, 0.9123328
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5040369, 0.5015600
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7298257, 0.7353165
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8130190, 0.8118577
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6557233, 0.6571193
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9092159, 0.9130495

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4611

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4611

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4161476, upper bound: 0.4177042
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174514, upper bound: 0.4164019
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8247566, 0.8255699
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7818978, 0.7796464
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7635050, 0.7634444
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9590020, 0.9579663
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9082181, 0.9105165
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5002353, 0.4972520
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7294207, 0.7244277
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8110301, 0.8122828
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6563680, 0.6562448
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9085670, 0.9049318

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4164006, upper bound: 0.4174526
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4174878, upper bound: 0.4163639
time: 3.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8247550, 0.8255715
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7818983, 0.7796460
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7635069, 0.7634423
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9590020, 0.9579661
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9082220, 0.9105127
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5002353, 0.4972520
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7294190, 0.7244294
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8110321, 0.8122811
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6563675, 0.6562450
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9085698, 0.9049289

Time for backsubstitution: 22.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4656

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4656

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4156153, upper bound: 0.4182361
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4167024, upper bound: 0.4171469
time: 3.13 seconds

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

Time for backsubstitution: 22.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177030, upper bound: 0.4161489
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4169198, upper bound: 0.4169343
time: 3.30 seconds

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

Time for backsubstitution: 22.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4187914, upper bound: 0.4150619
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4180072, upper bound: 0.4158471
time: 3.11 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4158459, upper bound: 0.4180083
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4171457, upper bound: 0.4167036
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4169330, upper bound: 0.4169210
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4182349, upper bound: 0.4156166
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4150606, upper bound: 0.4187926
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4163627, upper bound: 0.4174890
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4161476, upper bound: 0.4177042
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4174514, upper bound: 0.4164019
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4164006, upper bound: 0.4174526
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4174878, upper bound: 0.4163639
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4156153, upper bound: 0.4182361
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4167024, upper bound: 0.4171469
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4177030, upper bound: 0.4161489
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4169198, upper bound: 0.4169343
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.04
Output dim: 8, lower bound: -0.4187914, upper bound: 0.4150619
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.04
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

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1251
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 2812

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4158157, upper bound: 0.4150796
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4129029, upper bound: 0.4179781
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8240833, 0.8278601
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7812545, 0.7818323
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7628188, 0.7630346
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9630337, 0.9589999
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9100981, 0.9077580
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4976234, 0.4997370
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7240167, 0.7323203
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8117566, 0.8105421
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6544397, 0.6572714
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9046497, 0.9114029

Time for backsubstitution: 22.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 1251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1835

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4108400, upper bound: 0.4105116
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4109190, upper bound: 0.4104331
time: 3.45 seconds

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

Time for backsubstitution: 22.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1251
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 948

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4150484, upper bound: 0.4136999
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4137296, upper bound: 0.4150366
time: 3.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8248694, 0.8270741
1: -7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7812154, 0.7818714
2: -4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7633414, 0.7625124
3: -6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9630365, 0.9589970
4: -12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9096270, 0.9082291
5: -6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.4981825, 0.4991782
6: -5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7235715, 0.7327654
7: -11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8122706, 0.8100283
8: 9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6553953, 0.6563158
9: -7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9047956, 0.9112575

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2817
type: DSZ, layer: 3, pos: 1934
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2620
type: DSZ, layer: 3, pos: 1453
type: DSZ, layer: 3, pos: 1240
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 1403
type: DSZ, layer: 3, pos: 891
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 187
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 628
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 745
type: DSZ, layer: 3, pos: 977
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1943
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 235
type: DSZ, layer: 3, pos: 948
type: DSZ, layer: 3, pos: 905
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1251
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 2831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2817

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4166252, upper bound: 0.4021111
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4036547, upper bound: 0.4139802
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 22.68 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.45 + 564.54 = 621.98 seconds
