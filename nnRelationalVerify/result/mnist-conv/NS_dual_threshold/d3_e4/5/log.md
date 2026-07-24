## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.3532293525


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5333910, 2.5333900)
1: (-12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5630426, 2.5630426)
2: (-13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5471597, 2.5471601)
3: (-9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7141438, 2.7141433)
4: (-4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5921633, 1.5921636)
5: (-11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5612335, 2.5612335)
6: (-17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8651266, 2.8651257)
7: (-6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1917086, 2.1917083)
8: (-2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7734652, 1.7734652)
9: (2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2999578, 2.2999575)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.96 + 33.85 = 57.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.3600280, upper bound: 1.3600274

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586883, upper bound: 1.3577592
time: 4.64 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600241, upper bound: 1.3600237
time: 4.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.11 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 9.11
Output dim: 9, lower bound: -1.3586883, upper bound: 1.3577592
NS_B2, status: Status.UNKNOWN, split count: 1, time: 9.11
Output dim: 9, lower bound: -1.3600241, upper bound: 1.3600237

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -14.2945557, -10.3063278, -14.2906246, -10.3198032, -2.4956932, 2.5062785
1: -12.4920883, -8.9441929, -12.4902935, -8.9498053, -2.5432205, 2.5464072
2: -13.4024992, -10.1830997, -13.3974504, -10.1855955, -2.5293965, 2.5270209
3: -9.8843060, -6.9044428, -9.8801556, -6.9058075, -2.7035313, 2.7008681
4: -4.5604696, -2.4086146, -4.5602050, -2.4147823, -1.5711217, 1.5759726
5: -11.0682316, -7.3672581, -11.0646276, -7.3680811, -2.5447464, 2.5426569
6: -17.5765972, -13.6181316, -17.5740299, -13.6286755, -2.8310442, 2.8385425
7: -6.4314637, -3.5985951, -6.4302464, -3.6007948, -2.1809645, 2.1816576
8: -2.0365028, 0.1763811, -2.0340767, 0.1711984, -1.7557673, 1.7584820
9: 2.4228139, 5.1595278, 2.4267721, 5.1590314, -2.2886531, 2.2862980

Time for backsubstitution: 22.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586871, upper bound: 1.3563695
time: 5.81 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586871, upper bound: 1.3577557
time: 12.00 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -14.3000622, -10.2871847, -14.3440666, -10.2844639, -2.5217543, 2.5727825
1: -12.4945736, -8.9361744, -12.5052509, -8.9332790, -2.5575600, 2.5710230
2: -13.4097519, -10.1796169, -13.4136314, -10.1668425, -2.5583839, 2.5458698
3: -9.8902273, -6.9025431, -9.8948469, -6.8910999, -2.7245979, 2.7179208
4: -4.5608392, -2.3998163, -4.5728374, -2.3956418, -1.5898967, 1.5990360
5: -11.0733833, -7.3661013, -11.0779324, -7.3603792, -2.5599165, 2.5661564
6: -17.5802155, -13.6031580, -17.6217384, -13.6018171, -2.8512769, 2.9031134
7: -6.4332142, -3.5954447, -6.4381909, -3.5917931, -2.1951132, 2.1967556
8: -2.0398989, 0.1837716, -2.0595183, 0.1869984, -1.7740793, 1.7914858
9: 2.4171643, 5.1602302, 2.4132352, 5.1638827, -2.2996392, 2.3009441

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3586142
time: 4.78 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3600223
time: 4.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.29 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 9, lower bound: -1.3586871, upper bound: 1.3563695
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 9, lower bound: -1.3586871, upper bound: 1.3577557
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3586142
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3600223

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -14.2886181, -10.3067741, -14.2798958, -10.3206215, -2.4873157, 2.4940772
1: -12.4894543, -8.9464617, -12.4855137, -8.9540005, -2.5373745, 2.5391355
2: -13.3958855, -10.1869125, -13.3854208, -10.1927099, -2.5122118, 2.5052829
3: -9.8825283, -6.9177446, -9.8770161, -6.9295983, -2.6818924, 2.6865959
4: -4.5596962, -2.4103241, -4.5587463, -2.4179029, -1.5607991, 1.5659492
5: -11.0654345, -7.3767900, -11.0593300, -7.3853636, -2.5201092, 2.5253425
6: -17.5702114, -13.6187859, -17.5622883, -13.6298599, -2.8200312, 2.8227220
7: -6.4242210, -3.6014147, -6.4170828, -3.6060052, -2.1694922, 2.1684196
8: -2.0345559, 0.1727414, -2.0306230, 0.1645775, -1.7428689, 1.7485411
9: 2.4260039, 5.1590695, 2.4325786, 5.1581779, -2.2831888, 2.2787902

Time for backsubstitution: 23.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_B1_B1_B1

### Relational analysis result of NS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3496344, upper bound: 1.3560919
time: 7.66 seconds

## Relational analysis of NS_B1_B1_B2

### Relational analysis result of NS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3584046, upper bound: 1.3560892
time: 4.50 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -14.2945490, -10.3063278, -14.3016443, -10.3013592, -2.5221691, 2.5185757
1: -12.4920788, -8.9441957, -12.4927616, -8.9171944, -2.5858374, 2.5560822
2: -13.4024935, -10.1831036, -13.3994026, -10.1215334, -2.5748305, 2.5333619
3: -9.8843040, -6.9044609, -9.9334049, -6.8938169, -2.7170162, 2.7510145
4: -4.5604692, -2.4086182, -4.5677180, -2.4049468, -1.5769196, 1.5899925
5: -11.0682278, -7.3672786, -11.1272106, -7.3633347, -2.5448713, 2.6039169
6: -17.5765915, -13.6181326, -17.6098251, -13.6224461, -2.8636694, 2.8922486
7: -6.4314404, -3.5986004, -6.4394736, -3.5618877, -2.2241187, 2.1944621
8: -2.0365000, 0.1763716, -2.0576119, 0.1731687, -1.7673173, 1.7841783
9: 2.4228182, 5.1595268, 2.4189558, 5.1642027, -2.2943227, 2.2940826

Time for backsubstitution: 23.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_B1_B2_B1

### Relational analysis result of NS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3496344, upper bound: 1.3574708
time: 5.49 seconds

## Relational analysis of NS_B1_B2_B2

### Relational analysis result of NS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3584046, upper bound: 1.3574708
time: 5.54 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -14.2941246, -10.2876310, -14.3332958, -10.2852621, -2.5134258, 2.5604963
1: -12.4919405, -8.9384508, -12.5004339, -8.9373798, -2.5517750, 2.5635591
2: -13.4031439, -10.1835279, -13.4016132, -10.1743355, -2.5407615, 2.5241623
3: -9.8884544, -6.9158382, -9.8916569, -6.9152098, -2.7024517, 2.7036276
4: -4.5600486, -2.4015326, -4.5713196, -2.3987167, -1.5795398, 1.5889239
5: -11.0705853, -7.3756361, -11.0728340, -7.3777113, -2.5351305, 2.5484447
6: -17.5738354, -13.6038113, -17.6096687, -13.6029997, -2.8406444, 2.8866496
7: -6.4259758, -3.5982671, -6.4250579, -3.5969305, -2.1838608, 2.1836548
8: -2.0379572, 0.1801367, -2.0559006, 0.1803880, -1.7613206, 1.7813430
9: 2.4203510, 5.1597691, 2.4190249, 5.1630197, -2.2941461, 2.2934210

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_B2_B1_B1

### Relational analysis result of NS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509701, upper bound: 1.3583340
time: 5.14 seconds

## Relational analysis of NS_B2_B1_B2

### Relational analysis result of NS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597405, upper bound: 1.3583334
time: 5.65 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -14.3000555, -10.2871847, -14.3549137, -10.2660093, -2.5483627, 2.5815191
1: -12.4945650, -8.9361763, -12.5077095, -8.9004402, -2.6003618, 2.5805144
2: -13.4097443, -10.1796188, -13.4156008, -10.1035480, -2.5979843, 2.5522823
3: -9.8902264, -6.9025607, -9.9479723, -6.8794489, -2.7372403, 2.7679582
4: -4.5608401, -2.3998189, -4.5802431, -2.3856268, -1.5956244, 1.6071912
5: -11.0733805, -7.3661222, -11.1407986, -7.3556767, -2.5600371, 2.6228089
6: -17.5802040, -13.6031570, -17.6569729, -13.5955982, -2.8839741, 2.9355059
7: -6.4331923, -3.5954480, -6.4474144, -3.5527611, -2.2387791, 2.2097278
8: -2.0398970, 0.1837626, -2.0828872, 0.1889720, -1.7856927, 1.8076153
9: 2.4171672, 5.1602292, 2.4053874, 5.1690488, -2.3052878, 2.3086567

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of NS_B2_B2_B1

### Relational analysis result of NS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509701, upper bound: 1.3597396
time: 4.33 seconds

## Relational analysis of NS_B2_B2_B2

### Relational analysis result of NS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597405, upper bound: 1.3597396
time: 4.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.67 seconds
NS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3496344, upper bound: 1.3560919
NS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3584046, upper bound: 1.3560892
NS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3496344, upper bound: 1.3574708
NS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3584046, upper bound: 1.3574708
NS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3509701, upper bound: 1.3583340
NS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3597405, upper bound: 1.3583334
NS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3509701, upper bound: 1.3597396
NS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 31.67
Output dim: 9, lower bound: -1.3597405, upper bound: 1.3597396

## BFS NS instance: NS_B1_B1_B1

### Backsubstitution after applying NS history:
0: -14.2865295, -10.3074532, -14.2752571, -10.3249435, -2.4815683, 2.4887409
1: -12.4809914, -8.9479694, -12.4692917, -8.9608965, -2.5223475, 2.5217652
2: -13.3945513, -10.1875896, -13.3810930, -10.1954079, -2.5069947, 2.4978604
3: -9.8807936, -6.9181662, -9.8727856, -6.9312177, -2.6735239, 2.6816235
4: -4.5521002, -2.4119468, -4.5442276, -2.4259348, -1.5460696, 1.5502551
5: -11.0643177, -7.3786669, -11.0556011, -7.3901558, -2.5142360, 2.5176969
6: -17.5652370, -13.6202898, -17.5522251, -13.6359434, -2.8090639, 2.8105068
7: -6.4232516, -3.6054897, -6.4127760, -3.6144905, -2.1596384, 2.1577115
8: -2.0333071, 0.1700664, -2.0253696, 0.1595135, -1.7361536, 1.7402611
9: 2.4288282, 5.1518316, 2.4437876, 5.1446667, -2.2667713, 2.2602768

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of NS_B1_B1_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3482340, upper bound: 1.3560919
time: 6.85 seconds

## Relational analysis of NS_B1_B1_B1_A2

### Relational analysis result of NS_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3482340, upper bound: 1.3560881
time: 6.77 seconds

## BFS NS instance: NS_B1_B1_B2

### Backsubstitution after applying NS history:
0: -14.2886190, -10.3067770, -14.2798958, -10.3206224, -2.4869576, 2.4937353
1: -12.4894543, -8.9464588, -12.4855080, -8.9540033, -2.5373707, 2.5331922
2: -13.3958864, -10.1869125, -13.3854179, -10.1927128, -2.5106664, 2.5042682
3: -9.8825274, -6.9177465, -9.8770142, -6.9295993, -2.6871982, 2.6860833
4: -4.5596948, -2.4103260, -4.5587440, -2.4179044, -1.5607970, 1.5536263
5: -11.0654354, -7.3767929, -11.0593281, -7.3853664, -2.5199361, 2.5271306
6: -17.5702133, -13.6187859, -17.5622864, -13.6298618, -2.8200278, 2.8185406
7: -6.4242210, -3.6014171, -6.4170818, -3.6060095, -2.1610713, 2.1680789
8: -2.0345545, 0.1727414, -2.0306215, 0.1645756, -1.7395573, 1.7485387
9: 2.4260054, 5.1590676, 2.4325805, 5.1581750, -2.2771449, 2.2787864

Time for backsubstitution: 22.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of NS_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of NS_B1_B1_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3570050, upper bound: 1.3560893
time: 4.29 seconds

## Relational analysis of NS_B1_B1_B2_A2

### Relational analysis result of NS_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3570050, upper bound: 1.3560881
time: 5.06 seconds

## BFS NS instance: NS_B1_B2_B1

### Backsubstitution after applying NS history:
0: -14.2924519, -10.3070068, -14.2969646, -10.3056793, -2.5164204, 2.5132442
1: -12.4836121, -8.9457035, -12.4765320, -8.9240990, -2.5708179, 2.5387120
2: -13.4011612, -10.1837807, -13.3950863, -10.1241970, -2.5696473, 2.5259385
3: -9.8825684, -6.9048834, -9.9291849, -6.8954372, -2.7086496, 2.7443388
4: -4.5528731, -2.4102423, -4.5532002, -2.4129720, -1.5621901, 1.5742991
5: -11.0671091, -7.3691502, -11.1234894, -7.3681207, -2.5390015, 2.5966346
6: -17.5716114, -13.6196442, -17.5997410, -13.6285324, -2.8527040, 2.8800294
7: -6.4304714, -3.6026731, -6.4351730, -3.5703778, -2.2143285, 2.1837449
8: -2.0352535, 0.1736965, -2.0523562, 0.1680994, -1.7606006, 1.7746336
9: 2.4256444, 5.1522903, 2.4301720, 5.1506882, -2.2779050, 2.2755749

Time for backsubstitution: 23.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of NS_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of NS_B1_B2_B1_A1

### Relational analysis result of NS_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487959, upper bound: 1.3574711
time: 4.98 seconds

## Relational analysis of NS_B1_B2_B1_A2

### Relational analysis result of NS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487959, upper bound: 1.3574706
time: 4.42 seconds

## BFS NS instance: NS_B1_B2_B2

### Backsubstitution after applying NS history:
0: -14.2945471, -10.3063259, -14.3016434, -10.3013592, -2.5218172, 2.5182343
1: -12.4920778, -8.9441929, -12.4927578, -8.9171963, -2.5858340, 2.5501390
2: -13.4024925, -10.1831036, -13.3994026, -10.1215305, -2.5726876, 2.5323472
3: -9.8843040, -6.9044609, -9.9334040, -6.8938169, -2.7223191, 2.7489240
4: -4.5604668, -2.4086187, -4.5677137, -2.4049470, -1.5769165, 1.5776706
5: -11.0682278, -7.3672762, -11.1272097, -7.3633327, -2.5446973, 2.6040506
6: -17.5765915, -13.6181335, -17.6098213, -13.6224499, -2.8636684, 2.8875859
7: -6.4314394, -3.5986004, -6.4394732, -3.5618916, -2.2197688, 2.1941187
8: -2.0364990, 0.1763721, -2.0576110, 0.1731668, -1.7640061, 1.7828290
9: 2.4228187, 5.1595259, 2.4189596, 5.1641974, -2.2882786, 2.2940776

Time for backsubstitution: 23.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of NS_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of NS_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of NS_B1_B2_B2_A1

### Relational analysis result of NS_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3575668, upper bound: 1.3574707
time: 5.30 seconds

## Relational analysis of NS_B1_B2_B2_A2

### Relational analysis result of NS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3575668, upper bound: 1.3574738
time: 4.30 seconds

## BFS NS instance: NS_B2_B1_B1

### Backsubstitution after applying NS history:
0: -14.2919865, -10.2883110, -14.3285828, -10.2895746, -2.5076895, 2.5550058
1: -12.4834747, -8.9399643, -12.4841928, -8.9443254, -2.5367017, 2.5461783
2: -13.4018087, -10.1842003, -13.3973303, -10.1770477, -2.5355363, 2.5167170
3: -9.8867178, -6.9162598, -9.8874321, -6.9168267, -2.6940851, 2.6986570
4: -4.5524530, -2.4031527, -4.5567980, -2.4067564, -1.5648293, 1.5732331
5: -11.0694675, -7.3775110, -11.0691071, -7.3824978, -2.5292635, 2.5408182
6: -17.5688591, -13.6053190, -17.5995579, -13.6090927, -2.8296762, 2.8743796
7: -6.4250040, -3.6023502, -6.4207516, -3.6054564, -2.1739659, 2.1729431
8: -2.0367136, 0.1774626, -2.0506773, 0.1753235, -1.7546110, 1.7730923
9: 2.4231787, 5.1525326, 2.4303083, 5.1495018, -2.2777255, 2.2749531

Time for backsubstitution: 23.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of NS_B2_B1_B1_A1

### Relational analysis result of NS_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583336
time: 5.18 seconds

## Relational analysis of NS_B2_B1_B1_A2

### Relational analysis result of NS_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583333
time: 5.22 seconds

## BFS NS instance: NS_B2_B1_B2

### Backsubstitution after applying NS history:
0: -14.2941227, -10.2876320, -14.3332949, -10.2852631, -2.5131311, 2.5601578
1: -12.4919376, -8.9384489, -12.5004292, -8.9373798, -2.5517731, 2.5576153
2: -13.4031429, -10.1835270, -13.4016123, -10.1743393, -2.5392113, 2.5231462
3: -9.8884544, -6.9158382, -9.8916569, -6.9152107, -2.7077551, 2.7031155
4: -4.5600476, -2.4015326, -4.5713158, -2.3987184, -1.5795374, 1.5766034
5: -11.0705853, -7.3756347, -11.0728359, -7.3777142, -2.5349579, 2.5502338
6: -17.5738373, -13.6038094, -17.6096649, -13.6030006, -2.8406425, 2.8824654
7: -6.4259739, -3.5982685, -6.4250579, -3.5969353, -2.1754398, 2.1833119
8: -2.0379567, 0.1801362, -2.0559006, 0.1803856, -1.7580080, 1.7813404
9: 2.4203515, 5.1597681, 2.4190264, 5.1630149, -2.2881014, 2.2934182

Time for backsubstitution: 23.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of NS_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of NS_B2_B1_B2_A1

### Relational analysis result of NS_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583350, upper bound: 1.3583360
time: 4.53 seconds

## Relational analysis of NS_B2_B1_B2_A2

### Relational analysis result of NS_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583350, upper bound: 1.3583334
time: 5.20 seconds

## BFS NS instance: NS_B2_B2_B1

### Backsubstitution after applying NS history:
0: -14.2979155, -10.2878628, -14.3501587, -10.2703209, -2.5426254, 2.5760186
1: -12.4860973, -8.9376898, -12.4914589, -8.9074078, -2.5852957, 2.5631347
2: -13.4084063, -10.1802950, -13.4113283, -10.1062279, -2.5927944, 2.5448365
3: -9.8884916, -6.9029818, -9.9437532, -6.8810678, -2.7288761, 2.7612772
4: -4.5532441, -2.4014409, -4.5657234, -2.3936615, -1.5809147, 1.5914598
5: -11.0722647, -7.3679962, -11.1370773, -7.3604593, -2.5541749, 2.6155288
6: -17.5752296, -13.6046677, -17.6468563, -13.6016855, -2.8730049, 2.9232712
7: -6.4322224, -3.5995321, -6.4431081, -3.5612941, -2.2289386, 2.1990139
8: -2.0386577, 0.1810870, -2.0776615, 0.1839037, -1.7789841, 1.7980750
9: 2.4200001, 5.1529922, 2.4166732, 5.1555314, -2.2888680, 2.2901947

Time for backsubstitution: 23.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of NS_B2_B2_B1_A1

### Relational analysis result of NS_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3490729, upper bound: 1.3590329
time: 5.06 seconds

## Relational analysis of NS_B2_B2_B1_A2

### Relational analysis result of NS_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509662, upper bound: 1.3597359
time: 4.33 seconds

## BFS NS instance: NS_B2_B2_B2

### Backsubstitution after applying NS history:
0: -14.3000526, -10.2871838, -14.3549099, -10.2660103, -2.5480747, 2.5811765
1: -12.4945650, -8.9361753, -12.5077019, -8.9004421, -2.6003580, 2.5745707
2: -13.4097433, -10.1796179, -13.4156017, -10.1035519, -2.5958424, 2.5512662
3: -9.8902245, -6.9025612, -9.9479713, -6.8794489, -2.7425432, 2.7658675
4: -4.5608373, -2.3998199, -4.5802388, -2.3856270, -1.5956218, 1.5946168
5: -11.0733795, -7.3661222, -11.1407986, -7.3556747, -2.5598645, 2.6229503
6: -17.5802078, -13.6031609, -17.6569710, -13.5955963, -2.8839722, 2.9308448
7: -6.4331923, -3.5954504, -6.4474106, -3.5527658, -2.2344122, 2.2093847
8: -2.0398965, 0.1837621, -2.0828867, 0.1889710, -1.7823806, 1.8062692
9: 2.4171681, 5.1602278, 2.4053884, 5.1690459, -2.2992454, 2.3086536

Time for backsubstitution: 23.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of NS_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of NS_B2_B2_B2_A1

### Relational analysis result of NS_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578428, upper bound: 1.3590327
time: 4.79 seconds

## Relational analysis of NS_B2_B2_B2_A2

### Relational analysis result of NS_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597365, upper bound: 1.3597356
time: 4.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 38.64 seconds
NS_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3482340, upper bound: 1.3560919
NS_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3482340, upper bound: 1.3560881
NS_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3570050, upper bound: 1.3560893
NS_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3570050, upper bound: 1.3560881
NS_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3487959, upper bound: 1.3574711
NS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3487959, upper bound: 1.3574706
NS_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3575668, upper bound: 1.3574707
NS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3575668, upper bound: 1.3574738
NS_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583336
NS_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3495636, upper bound: 1.3583333
NS_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3583350, upper bound: 1.3583360
NS_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3583350, upper bound: 1.3583334
NS_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3490729, upper bound: 1.3590329
NS_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3509662, upper bound: 1.3597359
NS_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3578428, upper bound: 1.3590327
NS_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 38.64
Output dim: 9, lower bound: -1.3597365, upper bound: 1.3597356

## BFS NS instance: NS_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -14.2817574, -10.3078213, -14.2752571, -10.3249435, -2.4765806, 2.4876337
1: -12.4788408, -8.9498615, -12.4692917, -8.9608965, -2.5197039, 2.5204434
2: -13.3891411, -10.1910229, -13.3810930, -10.1954079, -2.4989457, 2.4944382
3: -9.8794432, -6.9286480, -9.8727856, -6.9312177, -2.6722441, 2.6729803
4: -4.5513935, -2.4133658, -4.5442276, -2.4259348, -1.5432529, 1.5471499
5: -11.0618076, -7.3864121, -11.0556011, -7.3901558, -2.5116754, 2.5079274
6: -17.5598240, -13.6208220, -17.5522251, -13.6359434, -2.8029261, 2.8089595
7: -6.4173298, -3.6078839, -6.4127760, -3.6144905, -2.1551247, 2.1549201
8: -2.0317450, 0.1670871, -2.0253696, 0.1595135, -1.7342138, 1.7354314
9: 2.4314327, 5.1514344, 2.4437876, 5.1446667, -2.2638626, 2.2594264

Time for backsubstitution: 23.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5816
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of NS_B1_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3474139, upper bound: 1.3560919
time: 4.73 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3474139, upper bound: 1.3560897
time: 6.56 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.81 + 551.76 = 609.57 seconds
