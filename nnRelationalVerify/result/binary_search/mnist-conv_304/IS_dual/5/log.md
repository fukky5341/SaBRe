## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.3532293525
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.8783393, 3.8783391)
1: (-12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.5584154, 3.5584154)
2: (-13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.2301512, 3.2301512)
3: (-9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981)
4: (-4.5608406, -2.3997998, -4.5608406, -2.3997998, -2.1610408, 2.1610408)
5: (-11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.7072897, 3.7072897)
6: (-17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.9770737, 3.9770737)
7: (-6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.8377733, 2.8377733)
8: (-2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.2236829, 2.2236829)
9: (2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.7430749, 2.7430749)

## BASE Result
execution time: IAR + LP analysis = 15.53 + 35.01 = 50.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1375822, upper bound: 2.1375790


# Binary Search by BASE starts (time budget: 3549.46 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.467238664627075
rel_dist={9: [-1.6640502761084588, 1.6640497405138106]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.360028225390102, 1.3600276268876046]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.188438653945923
rel_dist={9: [-1.0985895441039362, 1.0985873760165363]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.244198799133301
rel_dist={9: [-1.2428127078072388, 1.2428113778761434]}

## Binary Search Result
Binary search time: 208.56 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3340.90 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7526565, upper bound: 1.7504800
time: 5.80 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7535494, upper bound: 1.7535488
time: 4.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.61 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 10.61
Output dim: 9, lower bound: -1.7526565, upper bound: 1.7504800
IS_B2, status: Status.UNKNOWN, split count: 1, time: 10.61
Output dim: 9, lower bound: -1.7535494, upper bound: 1.7535488

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -14.2979565, -10.2945499, -14.2906237, -10.3198071, -3.0961733, 3.1160407
1: -12.4936256, -8.9392662, -12.4902925, -8.9498024, -3.0237913, 3.0297194
2: -13.4069471, -10.1809492, -13.3974476, -10.1855946, -3.0126238, 3.0081677
3: -9.8879452, -6.9032688, -9.8801556, -6.9058075, -2.9821377, 2.9768867
4: -4.5606976, -2.4032054, -4.5602050, -2.4147825, -1.8593855, 1.8684638
5: -11.0713968, -7.3665466, -11.0646286, -7.3680825, -3.0743756, 3.0704856
6: -17.5788231, -13.6089220, -17.5740318, -13.6286716, -3.3466902, 3.3606501
7: -6.4325342, -3.5966606, -6.4302454, -3.6007943, -2.5057259, 2.5070257
8: -2.0386004, 0.1809196, -2.0340772, 0.1711969, -2.0246868, 2.0298090
9: 2.4193392, 5.1599588, 2.4267726, 5.1590319, -2.5155468, 2.5111597

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504805, upper bound: 1.7504799
time: 4.64 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504805, upper bound: 1.7504799
time: 4.61 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -14.3000622, -10.2871742, -14.3440685, -10.2844639, -3.1270742, 3.1753473
1: -12.4945755, -8.9361677, -12.5052509, -8.9332781, -3.0390615, 3.0488248
2: -13.4097557, -10.1796150, -13.4136314, -10.1668415, -3.0362644, 3.0262957
3: -9.8902321, -6.9025416, -9.8948479, -6.8911009, -2.9991312, 2.9923062
4: -4.5608406, -2.3998075, -4.5728383, -2.3956420, -1.8809626, 1.8851926
5: -11.0733881, -7.3661022, -11.0779305, -7.3603792, -3.0846639, 3.0938377
6: -17.5802174, -13.6031513, -17.6217384, -13.6018181, -3.3713932, 3.4150949
7: -6.4332147, -3.5954432, -6.4381933, -3.5917931, -2.5179338, 2.5194609
8: -2.0399017, 0.1837773, -2.0595164, 0.1869969, -2.0421252, 2.0579019
9: 2.4171596, 5.1602306, 2.4132347, 5.1638832, -2.5226831, 2.5258389

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7535469, upper bound: 1.7511071
time: 4.09 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7535469, upper bound: 1.7535463
time: 4.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.47 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.7504805, upper bound: 1.7504799
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.7504805, upper bound: 1.7504799
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.7535469, upper bound: 1.7511071
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.7535469, upper bound: 1.7535463

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -14.2906237, -10.3198071, -14.2906237, -10.3198071, -3.0902967, 3.0902963
1: -12.4902925, -8.9498024, -12.4902925, -8.9498024, -3.0178728, 3.0178728
2: -13.3974476, -10.1855946, -13.3974476, -10.1855946, -3.0011835, 3.0011835
3: -9.8801556, -6.9058075, -9.8801556, -6.9058075, -2.9743481, 2.9743481
4: -4.5602050, -2.4147825, -4.5602050, -2.4147825, -1.8548455, 1.8548455
5: -11.0646286, -7.3680825, -11.0646286, -7.3680825, -3.0638776, 3.0638781
6: -17.5740318, -13.6286716, -17.5740318, -13.6286716, -3.3389311, 3.3389318
7: -6.4302454, -3.6007943, -6.4302454, -3.6007943, -2.5013266, 2.5013261
8: -2.0340772, 0.1711969, -2.0340772, 0.1711969, -2.0192776, 2.0192776
9: 2.4267726, 5.1590319, 2.4267726, 5.1590319, -2.5072742, 2.5072742

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504812, upper bound: 1.7480352
time: 4.22 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504812, upper bound: 1.7504781
time: 4.11 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -14.3440685, -10.2844639, -14.2906237, -10.3198071, -3.1420794, 3.1264887
1: -12.5052509, -8.9332781, -12.4902925, -8.9498024, -3.0335321, 3.0348520
2: -13.4136314, -10.1668415, -13.3974476, -10.1855946, -3.0196209, 3.0214558
3: -9.8948479, -6.8911009, -9.8801556, -6.9058075, -2.9890404, 2.9890547
4: -4.5728383, -2.3956420, -4.5602050, -2.4147825, -1.8676085, 1.8740499
5: -11.0779305, -7.3603792, -11.0646286, -7.3680825, -3.0778165, 3.0711136
6: -17.6217384, -13.6018181, -17.5740318, -13.6286716, -3.3870277, 3.3688343
7: -6.4381933, -3.5917931, -6.4302454, -3.6007943, -2.5120859, 2.5123434
8: -2.0595164, 0.1869969, -2.0340772, 0.1711969, -2.0442820, 2.0366664
9: 2.4132347, 5.1638832, 2.4267726, 5.1590319, -2.5206075, 2.5120029

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504834, upper bound: 1.7480346
time: 5.32 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504812, upper bound: 1.7504778
time: 4.27 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -14.2985582, -10.2872868, -14.3332977, -10.2852621, -3.1232901, 3.1639285
1: -12.4939117, -8.9367294, -12.5004339, -8.9373798, -3.0356040, 3.0425777
2: -13.4080896, -10.1805649, -13.4016132, -10.1743374, -3.0259514, 3.0075550
3: -9.8897762, -6.9059453, -9.8916588, -6.9152098, -2.9745665, 2.9857135
4: -4.5606484, -2.4002380, -4.5713191, -2.3987157, -1.8731375, 1.8779378
5: -11.0727081, -7.3685150, -11.0728359, -7.3777108, -3.0621285, 3.0850720
6: -17.5785885, -13.6033173, -17.6096687, -13.6030006, -3.3662715, 3.3999484
7: -6.4313869, -3.5961447, -6.4250574, -3.5969315, -2.5106850, 2.5088012
8: -2.0394130, 0.1828585, -2.0559011, 0.1803875, -2.0310364, 2.0520968
9: 2.4179645, 5.1601162, 2.4190240, 5.1630201, -2.5198445, 2.5191011

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7511050, upper bound: 1.7511071
time: 4.00 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7511050, upper bound: 1.7511048
time: 5.02 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -14.3000603, -10.2871742, -14.3549156, -10.2660084, -3.1536856, 3.1900640
1: -12.4945717, -8.9361706, -12.5077085, -8.9004412, -3.0818644, 3.0595288
2: -13.4097538, -10.1796160, -13.4156017, -10.1035480, -3.0817285, 3.0400505
3: -9.8902321, -6.9025517, -9.9479713, -6.8794494, -3.0107827, 3.0454197
4: -4.5608406, -2.3998094, -4.5802431, -2.3856273, -1.8866928, 1.9001368
5: -11.0733852, -7.3661103, -11.1407995, -7.3556747, -3.0940447, 3.1574659
6: -17.5802135, -13.6031542, -17.6569729, -13.5955944, -3.4040956, 3.4589527
7: -6.4332037, -3.5954461, -6.4474144, -3.5527616, -2.5666370, 2.5351188
8: -2.0399003, 0.1837716, -2.0828876, 0.1889720, -2.0574751, 2.0798995
9: 2.4171610, 5.1602316, 2.4053869, 5.1690493, -2.5283346, 2.5350518

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504804, upper bound: 1.7526542
time: 4.79 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7504781, upper bound: 1.7521049
time: 4.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.47 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7504812, upper bound: 1.7480352
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7504812, upper bound: 1.7504781
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7504834, upper bound: 1.7480346
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7504812, upper bound: 1.7504778
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7511050, upper bound: 1.7511071
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7511050, upper bound: 1.7511048
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7504804, upper bound: 1.7526542
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 9, lower bound: -1.7504781, upper bound: 1.7521049

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2891121, -10.3199186, -14.2798958, -10.3206205, -3.0864353, 3.0790420
1: -12.4896278, -8.9503593, -12.4855118, -8.9540005, -3.0143499, 3.0117722
2: -13.3957787, -10.1865044, -13.3854198, -10.1927109, -2.9913082, 2.9824095
3: -9.8796940, -6.9092159, -9.8770170, -6.9295988, -2.9500952, 2.9678011
4: -4.5600219, -2.4152117, -4.5587468, -2.4179029, -1.8470526, 1.8476803
5: -11.0639496, -7.3704996, -11.0593300, -7.3853621, -3.0414410, 3.0555062
6: -17.5723953, -13.6288414, -17.5622883, -13.6298618, -3.3337064, 3.3244312
7: -6.4284143, -3.6014938, -6.4170847, -3.6060047, -2.4938245, 2.4905353
8: -2.0335908, 0.1702785, -2.0306225, 0.1645775, -2.0080967, 2.0136757
9: 2.4275799, 5.1589184, 2.4325776, 5.1581783, -2.5044560, 2.5005558

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7480381, upper bound: 1.7480384
time: 4.10 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7480381, upper bound: 1.7480378
time: 4.60 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2906208, -10.3198061, -14.3016462, -10.3013592, -3.1167746, 3.1051102
1: -12.4902906, -8.9498043, -12.4927616, -8.9171915, -3.0604897, 3.0287623
2: -13.3974457, -10.1855946, -13.3994026, -10.1215305, -3.0537062, 3.0148687
3: -9.8801546, -6.9058161, -9.9334059, -6.8938150, -2.9863396, 3.0275898
4: -4.5602045, -2.4147844, -4.5677190, -2.4049456, -1.8606458, 1.8711641
5: -11.0646257, -7.3680944, -11.1272087, -7.3633318, -3.0732613, 3.1282396
6: -17.5740280, -13.6286736, -17.6098251, -13.6224480, -3.3715620, 3.3962049
7: -6.4302316, -3.6007977, -6.4394741, -3.5618873, -2.5497291, 2.5168152
8: -2.0340757, 0.1711926, -2.0576124, 0.1731682, -2.0345626, 2.0498960
9: 2.4267750, 5.1590309, 2.4189563, 5.1642022, -2.5129447, 2.5165572

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501680, upper bound: 1.7288198
time: 4.70 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501680, upper bound: 1.7501677
time: 4.03 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3425550, -10.2845745, -14.2798958, -10.3206205, -3.1381969, 3.1152453
1: -12.5045795, -8.9338226, -12.4855118, -8.9540005, -3.0299940, 3.0287595
2: -13.4119635, -10.1678028, -13.3854198, -10.1927109, -3.0097504, 3.0026240
3: -9.8943806, -6.8945432, -9.8770170, -6.9295988, -2.9647818, 2.9824739
4: -4.5726461, -2.3960640, -4.5587468, -2.4179029, -1.8598032, 1.8668973
5: -11.0772743, -7.3627987, -11.0593300, -7.3853621, -3.0553989, 3.0627356
6: -17.6200638, -13.6019840, -17.5622883, -13.6298618, -3.3817334, 3.3543456
7: -6.4363651, -3.5924854, -6.4170847, -3.6060047, -2.5046091, 2.5015693
8: -2.0590134, 0.1860800, -2.0306225, 0.1645775, -2.0330739, 2.0310757
9: 2.4140396, 5.1637683, 2.4325776, 5.1581783, -2.5177913, 2.5052824

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7502112, upper bound: 1.7480366
time: 4.26 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7502112, upper bound: 1.7480373
time: 4.38 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.3440647, -10.2844639, -14.3016462, -10.3013592, -3.1530342, 3.1413031
1: -12.5052471, -8.9332771, -12.4927616, -8.9171915, -3.0761509, 3.0457411
2: -13.4136286, -10.1668453, -13.3994026, -10.1215305, -3.0688667, 3.0351415
3: -9.8948460, -6.8911123, -9.9334059, -6.8938150, -3.0010309, 3.0422935
4: -4.5728388, -2.3956432, -4.5677190, -2.4049456, -1.8734088, 1.8903694
5: -11.0779324, -7.3603873, -11.1272087, -7.3633318, -3.0872011, 3.1354756
6: -17.6217384, -13.6018181, -17.6098251, -13.6224480, -3.4165368, 3.4261065
7: -6.4381828, -3.5917945, -6.4394741, -3.5618873, -2.5601194, 2.5278335
8: -2.0595164, 0.1869926, -2.0576124, 0.1731682, -2.0588851, 2.0657051
9: 2.4132371, 5.1638827, 2.4189563, 5.1642022, -2.5262799, 2.5212855

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7523489, upper bound: 1.7288166
time: 4.59 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7523489, upper bound: 1.7501669
time: 4.18 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -14.2893524, -10.2879877, -14.3332977, -10.2852621, -3.1137180, 3.1618705
1: -12.4897919, -8.9402561, -12.5004339, -8.9373798, -3.0306354, 3.0399690
2: -13.3977442, -10.1870441, -13.4016132, -10.1743374, -3.0105934, 3.0013084
3: -9.8871155, -6.9263172, -9.8916588, -6.9152098, -2.9719057, 2.9653416
4: -4.5593257, -2.4029460, -4.5713191, -2.3987157, -1.8677902, 1.8719811
5: -11.0680752, -7.3833766, -11.0728359, -7.3777108, -3.0572543, 3.0663595
6: -17.5683327, -13.6043339, -17.6096687, -13.6030006, -3.3546557, 3.3971236
7: -6.4200583, -3.6006594, -6.4250574, -3.5969315, -2.5021687, 2.5035970
8: -2.0364175, 0.1771660, -2.0559011, 0.1803875, -2.0274768, 2.0429311
9: 2.4229479, 5.1593742, 2.4190240, 5.1630201, -2.5142765, 2.5174737

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7502112
time: 4.27 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7496621
time: 5.00 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -14.3110447, -10.2687159, -14.3332977, -10.2852621, -3.1348600, 3.1753833
1: -12.4970131, -8.9031792, -12.5004339, -8.9373798, -3.0406346, 3.0853848
2: -13.4117184, -10.1155996, -13.4016132, -10.1743374, -3.0255175, 3.0612054
3: -9.9435444, -6.8909054, -9.8916588, -6.9152098, -3.0283346, 3.0007534
4: -4.5683641, -2.3898175, -4.5713191, -2.3987157, -1.8775992, 1.8848593
5: -11.1362381, -7.3613510, -11.0728359, -7.3777108, -3.1270700, 3.0881457
6: -17.6162891, -13.5969276, -17.6096687, -13.6030006, -3.4204335, 3.4303298
7: -6.4423928, -3.5563617, -6.4250574, -3.5969315, -2.5252643, 2.5580139
8: -2.0637131, 0.1857500, -2.0559011, 0.1803875, -2.0626392, 2.0525928
9: 2.4093256, 5.1654458, 2.4190240, 5.1630201, -2.5280280, 2.5251129

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7502108
time: 4.77 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7496628
time: 4.40 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -14.2906208, -10.3198061, -14.3549156, -10.2660084, -3.1530871, 3.1567955
1: -12.4902906, -8.9498043, -12.5077085, -8.9004412, -3.0776539, 3.0442381
2: -13.3974457, -10.1855946, -13.4156017, -10.1035480, -3.0669103, 3.0333748
3: -9.8801546, -6.9058161, -9.9479713, -6.8794494, -3.0007052, 3.0421553
4: -4.5602045, -2.4147844, -4.5802431, -2.3856273, -1.8801272, 1.8824046
5: -11.0646257, -7.3680944, -11.1407995, -7.3556747, -3.0804949, 3.1424465
6: -17.5740280, -13.6286736, -17.6569729, -13.5955944, -3.4015274, 3.4308739
7: -6.4302316, -3.6007977, -6.4474144, -3.5527616, -2.5607166, 2.5277431
8: -2.0340757, 0.1711926, -2.0828876, 0.1889720, -2.0520153, 2.0662582
9: 2.4267750, 5.1590309, 2.4053869, 5.1690493, -2.5176530, 2.5299492

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7288172, upper bound: 1.7523481
time: 4.64 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501648, upper bound: 1.7523484
time: 4.19 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -14.3440647, -10.2844639, -14.3549156, -10.2660084, -3.1727629, 3.1608706
1: -12.5052471, -8.9332771, -12.5077085, -8.9004412, -3.0873766, 3.0555849
2: -13.4136286, -10.1668453, -13.4156017, -10.1035480, -3.0844550, 3.0451059
3: -9.8948460, -6.8911123, -9.9479713, -6.8794494, -3.0153966, 3.0568590
4: -4.5728388, -2.3956432, -4.5802431, -2.3856273, -1.8880198, 1.8986435
5: -11.0779324, -7.3603873, -11.1407995, -7.3556747, -3.1051826, 3.1594582
6: -17.6217384, -13.6018181, -17.6569729, -13.5955944, -3.4197674, 3.4447217
7: -6.4381828, -3.5917945, -6.4474144, -3.5527616, -2.5720463, 2.5389652
8: -2.0595164, 0.1869926, -2.0828876, 0.1889720, -2.0700703, 2.0821340
9: 2.4132371, 5.1638827, 2.4053869, 5.1690493, -2.5337400, 2.5372310

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501650, upper bound: 1.7304411
time: 4.01 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501650, upper bound: 1.7518331
time: 4.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.46 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7480381, upper bound: 1.7480384
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7480381, upper bound: 1.7480378
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7501680, upper bound: 1.7288198
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7501680, upper bound: 1.7501677
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7502112, upper bound: 1.7480366
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7502112, upper bound: 1.7480373
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7523489, upper bound: 1.7288166
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7523489, upper bound: 1.7501669
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7502112
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7496621
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7502108
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7480365, upper bound: 1.7496628
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7288172, upper bound: 1.7523481
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7501648, upper bound: 1.7523484
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7501650, upper bound: 1.7304411
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 9, lower bound: -1.7501650, upper bound: 1.7518331

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206205, -14.2798958, -10.3206205, -3.0769920, 3.0769920
1: -12.4855118, -8.9540005, -12.4855118, -8.9540005, -3.0093851, 3.0093846
2: -13.3854198, -10.1927109, -13.3854198, -10.1927109, -2.9759502, 2.9759502
3: -9.8770170, -6.9295988, -9.8770170, -6.9295988, -2.9474182, 2.9474182
4: -4.5587468, -2.4179029, -4.5587468, -2.4179029, -1.8417072, 1.8417075
5: -11.0593300, -7.3853621, -11.0593300, -7.3853621, -3.0367908, 3.0367899
6: -17.5622883, -13.6298618, -17.5622883, -13.6298618, -3.3215547, 3.3215549
7: -6.4170847, -3.6060047, -6.4170847, -3.6060047, -2.4852743, 2.4852743
8: -2.0306225, 0.1645775, -2.0306225, 0.1645775, -2.0045047, 2.0045049
9: 2.4325776, 5.1581783, 2.4325776, 5.1581783, -2.4989033, 2.4989033

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263796, upper bound: 1.7477352
time: 4.20 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477343, upper bound: 1.7477330
time: 4.93 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.3016462, -10.3013592, -14.2798958, -10.3206205, -3.0981750, 3.1058311
1: -12.4927616, -8.9171915, -12.4855118, -8.9540005, -3.0192537, 3.0547600
2: -13.3994026, -10.1215305, -13.3854198, -10.1927109, -2.9909363, 3.0358343
3: -9.9334059, -6.8938150, -9.8770170, -6.9295988, -3.0038071, 2.9832020
4: -4.5677190, -2.4049456, -4.5587468, -2.4179029, -1.8514714, 1.8544576
5: -11.1272087, -7.3633318, -11.0593300, -7.3853621, -3.1065073, 3.0586004
6: -17.6098251, -13.6224480, -17.5622883, -13.6298618, -3.3868356, 3.3575027
7: -6.4394741, -3.5618873, -6.4170847, -3.6060047, -2.5085459, 2.5396292
8: -2.0576124, 0.1731682, -2.0306225, 0.1645775, -2.0392776, 2.0142488
9: 2.4189563, 5.1642022, 2.4325776, 5.1581783, -2.5127101, 2.5065036

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477345, upper bound: 1.7263805
time: 4.29 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477345, upper bound: 1.7477329
time: 4.82 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2859602, -10.3241262, -14.3005657, -10.3016987, -3.1119347, 3.1000667
1: -12.4740620, -8.9566898, -12.4885006, -8.9179525, -3.0437303, 3.0178614
2: -13.3931007, -10.1882935, -13.3987370, -10.1218615, -3.0467539, 3.0103135
3: -9.8759270, -6.9074383, -9.9325352, -6.8940296, -2.9818974, 3.0250969
4: -4.5456858, -2.4228182, -4.5638962, -2.4057584, -1.8456123, 1.8599427
5: -11.0608978, -7.3728833, -11.1266499, -7.3642778, -3.0671329, 3.1229062
6: -17.5639420, -13.6347523, -17.6073227, -13.6232023, -3.3600659, 3.3876214
7: -6.4259291, -3.6092806, -6.4389868, -3.5639372, -2.5410299, 2.5075169
8: -2.0288219, 0.1661272, -2.0569849, 0.1718221, -2.0276623, 2.0438867
9: 2.4379826, 5.1455183, 2.4203706, 5.1605606, -2.4980443, 2.5015574

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477329, upper bound: 1.7288202
time: 4.20 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477331, upper bound: 1.7263805
time: 4.29 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.2906170, -10.3198051, -14.3016462, -10.3013592, -3.1169567, 3.1049533
1: -12.4902849, -8.9498043, -12.4927616, -8.9171915, -3.0556793, 3.0287604
2: -13.3974457, -10.1855946, -13.3994026, -10.1215305, -3.0559440, 3.0138597
3: -9.8801517, -6.9058189, -9.9334059, -6.8938150, -2.9863367, 3.0275869
4: -4.5602012, -2.4147863, -4.5677190, -2.4049456, -1.8506756, 1.8711638
5: -11.0646267, -7.3680935, -11.1272087, -7.3633318, -3.0753822, 3.1282377
6: -17.5740204, -13.6286755, -17.6098251, -13.6224480, -3.3681746, 3.3962040
7: -6.4302320, -3.6008024, -6.4394741, -3.5618873, -2.5475881, 2.5100956
8: -2.0340748, 0.1711907, -2.0576124, 0.1731682, -2.0345607, 2.0472147
9: 2.4267774, 5.1590276, 2.4189563, 5.1642022, -2.5129423, 2.5116653

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477331, upper bound: 1.7501675
time: 4.46 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477331, upper bound: 1.7477329
time: 4.91 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.3332977, -10.2852621, -14.2798958, -10.3206205, -3.1286125, 3.1132784
1: -12.5004339, -8.9373798, -12.4855118, -8.9540005, -3.0249319, 3.0264297
2: -13.4016132, -10.1743374, -13.3854198, -10.1927109, -2.9944267, 2.9957857
3: -9.8916588, -6.9152098, -9.8770170, -6.9295988, -2.9620600, 2.9618073
4: -4.5713191, -2.3987157, -4.5587468, -2.4179029, -1.8543782, 1.8610036
5: -11.0728359, -7.3777108, -11.0593300, -7.3853621, -3.0509129, 3.0439644
6: -17.6096687, -13.6030006, -17.5622883, -13.6298618, -3.3690009, 3.3515544
7: -6.4250574, -3.5969315, -6.4170847, -3.6060047, -2.4961643, 2.4964323
8: -2.0559011, 0.1803875, -2.0306225, 0.1645775, -2.0293045, 2.0219812
9: 2.4190240, 5.1630201, 2.4325776, 5.1581783, -2.5122514, 2.5036125

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7263751
time: 4.29 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7477299
time: 4.31 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.3549156, -10.2660084, -14.2798958, -10.3206205, -3.1497517, 3.1421432
1: -12.5077085, -8.9004412, -12.4855118, -8.9540005, -3.0347319, 3.0719233
2: -13.4156017, -10.1035480, -13.3854198, -10.1927109, -3.0094490, 3.0490379
3: -9.9479713, -6.8794494, -9.8770170, -6.9295988, -3.0183725, 2.9975677
4: -4.5802431, -2.3856273, -4.5587468, -2.4179029, -1.8640904, 1.8739388
5: -11.1407995, -7.3556747, -11.0593300, -7.3853621, -3.1207132, 3.0658073
6: -17.6569729, -13.5955944, -17.5622883, -13.6298618, -3.4254999, 3.3874688
7: -6.4474144, -3.5527616, -6.4170847, -3.6060047, -2.5194802, 2.5506167
8: -2.0828876, 0.1889720, -2.0306225, 0.1645775, -2.0557227, 2.0317907
9: 2.4053869, 5.1690493, 2.4325776, 5.1581783, -2.5260811, 2.5112119

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7263750
time: 4.98 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7477301
time: 5.19 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.3393250, -10.2887726, -14.3005657, -10.3016987, -3.1480322, 3.1362662
1: -12.4890041, -8.9402256, -12.4885006, -8.9179525, -3.0593753, 3.0347939
2: -13.4093246, -10.1695538, -13.3987370, -10.1218615, -3.0618963, 3.0305820
3: -9.8906221, -6.8927283, -9.9325352, -6.8940296, -2.9965925, 3.0398068
4: -4.5583167, -2.4036858, -4.5638962, -2.4057584, -1.8583729, 1.8791203
5: -11.0742016, -7.3651714, -11.1266499, -7.3642778, -3.0810642, 3.1301465
6: -17.6116180, -13.6079102, -17.6073227, -13.6232023, -3.4050822, 3.4175172
7: -6.4338789, -3.6003213, -6.4389868, -3.5639372, -2.5514140, 2.5184731
8: -2.0542908, 0.1819282, -2.0569849, 0.1718221, -2.0505090, 2.0596859
9: 2.4245195, 5.1503639, 2.4203706, 5.1605606, -2.5113032, 2.5062768

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7288170
time: 4.12 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7263750
time: 4.75 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.3440609, -10.2844677, -14.3016462, -10.3013592, -3.1531601, 3.1411433
1: -12.5052404, -8.9332800, -12.4927616, -8.9171915, -3.0713387, 3.0457401
2: -13.4136267, -10.1668453, -13.3994026, -10.1215305, -3.0711083, 3.0341296
3: -9.8948450, -6.8911119, -9.9334059, -6.8938150, -3.0010300, 3.0422940
4: -4.5728331, -2.3956444, -4.5677190, -2.4049456, -1.8634377, 1.8903673
5: -11.0779285, -7.3603902, -11.1272087, -7.3633318, -3.0893211, 3.1354742
6: -17.6217327, -13.6018200, -17.6098251, -13.6224480, -3.4131489, 3.4261060
7: -6.4381838, -3.5917974, -6.4394741, -3.5618873, -2.5579784, 2.5211129
8: -2.0595150, 0.1869907, -2.0576124, 0.1731682, -2.0575385, 2.0630243
9: 2.4132390, 5.1638789, 2.4189563, 5.1642022, -2.5262766, 2.5163941

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7501644
time: 4.39 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7477303
time: 5.03 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206205, -14.3332977, -10.2852621, -3.1132784, 3.1286125
1: -12.4855118, -8.9540005, -12.5004339, -8.9373798, -3.0264292, 3.0249310
2: -13.3854198, -10.1927109, -13.4016132, -10.1743374, -2.9957848, 2.9944263
3: -9.8770170, -6.9295988, -9.8916588, -6.9152098, -2.9618073, 2.9620600
4: -4.5587468, -2.4179029, -4.5713191, -2.3987157, -1.8610034, 1.8543782
5: -11.0593300, -7.3853621, -11.0728359, -7.3777108, -3.0439649, 3.0509124
6: -17.5622883, -13.6298618, -17.6096687, -13.6030006, -3.3515544, 3.3690012
7: -6.4170847, -3.6060047, -6.4250574, -3.5969315, -2.4964323, 2.4961646
8: -2.0306225, 0.1645775, -2.0559011, 0.1803875, -2.0219812, 2.0293047
9: 2.4325776, 5.1581783, 2.4190240, 5.1630201, -2.5036125, 2.5122509

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263765, upper bound: 1.7499105
time: 7.46 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477313, upper bound: 1.7499110
time: 4.16 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -14.3332977, -10.2852621, -14.3332977, -10.2852621, -3.1327801, 3.1327806
1: -12.5004339, -8.9373798, -12.5004339, -8.9373798, -3.0362601, 3.0362601
2: -13.4016132, -10.1743374, -13.4016132, -10.1743374, -3.0064163, 3.0064163
3: -9.8916588, -6.9152098, -9.8916588, -6.9152098, -2.9764490, 2.9764490
4: -4.5713191, -2.3987157, -4.5713191, -2.3987157, -1.8691506, 1.8691506
5: -11.0728359, -7.3777108, -11.0728359, -7.3777108, -3.0683093, 3.0683098
6: -17.6096687, -13.6030006, -17.6096687, -13.6030006, -3.3700452, 3.3700449
7: -6.4250574, -3.5969315, -6.4250574, -3.5969315, -2.5075269, 2.5075269
8: -2.0559011, 0.1803875, -2.0559011, 0.1803875, -2.0398307, 2.0398312
9: 2.4190240, 5.1630201, 2.4190240, 5.1630201, -2.5196543, 2.5196543

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477315, upper bound: 1.7279986
time: 4.63 seconds

## Relational analysis of IS_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477315, upper bound: 1.7493910
time: 4.76 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -14.3016462, -10.3013592, -14.3332977, -10.2852621, -3.1344614, 3.1420062
1: -12.4927616, -8.9171915, -12.5004339, -8.9373798, -3.0362988, 3.0703063
2: -13.3994026, -10.1215305, -13.4016132, -10.1743374, -3.0107708, 3.0510187
3: -9.9334059, -6.8938150, -9.8916588, -6.9152098, -3.0181961, 2.9978437
4: -4.5677190, -2.4049456, -4.5713191, -2.3987157, -1.8707676, 1.8671284
5: -11.1272087, -7.3633318, -11.0728359, -7.3777108, -3.1136823, 3.0727229
6: -17.6098251, -13.6224480, -17.6096687, -13.6030006, -3.4168363, 3.4023042
7: -6.4394741, -3.5618873, -6.4250574, -3.5969315, -2.5197043, 2.5501072
8: -2.0576124, 0.1731682, -2.0559011, 0.1803875, -2.0552235, 2.0390487
9: 2.4189563, 5.1642022, 2.4190240, 5.1630201, -2.5174189, 2.5198512

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7288171, upper bound: 1.7499095
time: 4.54 seconds

## Relational analysis of IS_B2_B1_A2_A1_B2

### Relational analysis result of IS_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501647, upper bound: 1.7499098
time: 4.82 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -14.3549156, -10.2660084, -14.3332977, -10.2852621, -3.1539173, 3.1616583
1: -12.5077085, -8.9004412, -12.5004339, -8.9373798, -3.0461450, 3.0817566
2: -13.4156017, -10.1035480, -13.4016132, -10.1743374, -3.0214386, 3.0666237
3: -9.9479713, -6.8794494, -9.8916588, -6.9152098, -3.0327616, 3.0122094
4: -4.5802431, -2.3856273, -4.5713191, -2.3987157, -1.8789201, 1.8818626
5: -11.1407995, -7.3556747, -11.0728359, -7.3777108, -3.1376891, 3.0901012
6: -17.6569729, -13.5955944, -17.6096687, -13.6030006, -3.4354520, 3.4059680
7: -6.4474144, -3.5527616, -6.4250574, -3.5969315, -2.5308428, 2.5620806
8: -2.0828876, 0.1889720, -2.0559011, 0.1803875, -2.0716815, 2.0496407
9: 2.4053869, 5.1690493, 2.4190240, 5.1630201, -2.5333924, 2.5272751

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_B2_B1_A2_A2_A1

### Relational analysis result of IS_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501649, upper bound: 1.7279970
time: 5.42 seconds

## Relational analysis of IS_B2_B1_A2_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7501649, upper bound: 1.7493891
time: 4.73 seconds

## BFS IS instance: IS_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2895374, -10.3201447, -14.3501587, -10.2703190, -3.1480474, 3.1519804
1: -12.4860296, -8.9505606, -12.4914598, -8.9074078, -3.0667105, 3.0274644
2: -13.3967781, -10.1859341, -13.4113274, -10.1062279, -3.0623693, 3.0263758
3: -9.8792830, -6.9060278, -9.9437532, -6.8810663, -2.9982166, 3.0377254
4: -4.5563812, -2.4155989, -4.5657234, -2.3936605, -1.8688798, 1.8673348
5: -11.0640650, -7.3690376, -11.1370764, -7.3604593, -3.0751657, 3.1363087
6: -17.5715218, -13.6294327, -17.6468582, -13.6016884, -3.3929315, 3.4194062
7: -6.4297481, -3.6028466, -6.4431086, -3.5612941, -2.5514259, 2.5198696
8: -2.0334473, 0.1698470, -2.0776596, 0.1839032, -2.0460095, 2.0578856
9: 2.4281902, 5.1553907, 2.4166737, 5.1555319, -2.5026438, 2.5149782

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B2_B2_A1_B1_A1

### Relational analysis result of IS_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7523485
time: 4.30 seconds

## Relational analysis of IS_B2_B2_A1_B1_A2

### Relational analysis result of IS_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7499121
time: 4.03 seconds

## BFS IS instance: IS_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2906208, -10.3198061, -14.3549118, -10.2660093, -3.1529279, 3.1569777
1: -12.4902906, -8.9498043, -12.5077019, -8.9004421, -3.0776520, 3.0394268
2: -13.3974457, -10.1855946, -13.4156017, -10.1035509, -3.0654984, 3.0337844
3: -9.8801546, -6.9058161, -9.9479713, -6.8794489, -3.0007057, 3.0421553
4: -4.5602045, -2.4147844, -4.5802383, -2.3856285, -1.8801265, 1.8724163
5: -11.0646257, -7.3680944, -11.1407967, -7.3556747, -3.0804939, 3.1445689
6: -17.5740280, -13.6286736, -17.6569691, -13.5955963, -3.4015265, 3.4274864
7: -6.4302316, -3.6007977, -6.4474111, -3.5527673, -2.5579371, 2.5274017
8: -2.0340757, 0.1711926, -2.0828876, 0.1889706, -2.0493345, 2.0649126
9: 2.4267750, 5.1590309, 2.4053893, 5.1690454, -2.5127611, 2.5299468

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of IS_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B2_B2_A1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477298, upper bound: 1.7523484
time: 4.41 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7477298, upper bound: 1.7499111
time: 4.31 seconds

## BFS IS instance: IS_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -14.3393250, -10.2887726, -14.3538342, -10.2663488, -3.1679454, 3.1558380
1: -12.4890041, -8.9402256, -12.5034409, -8.9012098, -3.0705948, 3.0446377
2: -13.4093246, -10.1695538, -13.4149284, -10.1038828, -3.0774813, 3.0405474
3: -9.8906221, -6.8927283, -9.9471035, -6.8796601, -3.0109620, 3.0543752
4: -4.5583167, -2.4036858, -4.5764217, -2.3864431, -1.8729911, 1.8874421
5: -11.0742016, -7.3651714, -11.1402359, -7.3566170, -3.0990734, 3.1541314
6: -17.6116180, -13.6079102, -17.6544647, -13.5963535, -3.4082723, 3.4361329
7: -6.4338789, -3.6003213, -6.4469261, -3.5548220, -2.5633268, 2.5296040
8: -2.0542908, 0.1819282, -2.0822692, 0.1876259, -2.0632014, 2.0761178
9: 2.4245195, 5.1503639, 2.4068198, 5.1654077, -2.5188832, 2.5222380

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7304392
time: 4.30 seconds

## Relational analysis of IS_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7279987
time: 4.35 seconds

## BFS IS instance: IS_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -14.3440609, -10.2844677, -14.3549156, -10.2660084, -3.1729441, 3.1607103
1: -12.5052404, -8.9332800, -12.5077085, -8.9004412, -3.0825648, 3.0555849
2: -13.4136267, -10.1668453, -13.4156017, -10.1035480, -3.0867090, 3.0440936
3: -9.8948450, -6.8911119, -9.9479713, -6.8794494, -3.0153956, 3.0568595
4: -4.5728331, -2.3956444, -4.5802431, -2.3856273, -1.8780477, 1.8986416
5: -11.0779285, -7.3603902, -11.1407995, -7.3556747, -3.1073027, 3.1594567
6: -17.6217327, -13.6018200, -17.6569729, -13.5955944, -3.4163790, 3.4447207
7: -6.4381838, -3.5917974, -6.4474144, -3.5527616, -2.5699053, 2.5322444
8: -2.0595150, 0.1869907, -2.0828876, 0.1889720, -2.0700693, 2.0794535
9: 2.4132390, 5.1638789, 2.4053869, 5.1690493, -2.5337372, 2.5323374

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7518334
time: 4.94 seconds

## Relational analysis of IS_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7493891
time: 5.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.53 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7263796, upper bound: 1.7477352
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477343, upper bound: 1.7477330
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477345, upper bound: 1.7263805
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477345, upper bound: 1.7477329
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477329, upper bound: 1.7288202
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477331, upper bound: 1.7263805
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477331, upper bound: 1.7501675
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477331, upper bound: 1.7477329
IS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7263751
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7477299
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7263750
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499112, upper bound: 1.7477301
IS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7288170
IS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7263750
IS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7501644
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7499099, upper bound: 1.7477303
IS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7263765, upper bound: 1.7499105
IS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477313, upper bound: 1.7499110
IS_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477315, upper bound: 1.7279986
IS_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477315, upper bound: 1.7493910
IS_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7288171, upper bound: 1.7499095
IS_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7501647, upper bound: 1.7499098
IS_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7501649, upper bound: 1.7279970
IS_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7501649, upper bound: 1.7493891
IS_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7523485
IS_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7263752, upper bound: 1.7499121
IS_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477298, upper bound: 1.7523484
IS_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7477298, upper bound: 1.7499111
IS_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7304392
IS_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7279987
IS_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7518334
IS_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 9, lower bound: -1.7508166, upper bound: 1.7493891
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.522998809814453
rel_dist={9: [-1.7535540543698414, 1.753553758870325]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655946, upper bound: 1.4669623
time: 4.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4681558, upper bound: 1.4681563
time: 9.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.74
Output dim: 9, lower bound: -1.4655946, upper bound: 1.4669623
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.74
Output dim: 9, lower bound: -1.4681558, upper bound: 1.4681563

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2906237, -10.3198071, -14.2955589, -10.3028746, -2.6592398, 2.6459341
1: -12.4902925, -8.9498024, -12.4925385, -8.9427500, -2.6674762, 2.6634822
2: -13.3974476, -10.1855946, -13.4037971, -10.1824665, -2.6474471, 2.6504312
3: -9.8801556, -6.9058075, -9.8853703, -6.9040971, -2.7708688, 2.7742195
4: -4.5602050, -2.4147825, -4.5605364, -2.4070325, -1.6493719, 1.6432791
5: -11.0646286, -7.3680825, -11.0691547, -7.3670487, -2.6747475, 2.6773663
6: -17.5740318, -13.6286716, -17.5772495, -13.6154308, -2.9695077, 2.9601064
7: -6.4302454, -3.6007943, -6.4317760, -3.5980282, -2.2631154, 2.2622433
8: -2.0340772, 0.1711969, -2.0371189, 0.1777091, -1.8265243, 1.8231087
9: 2.4267726, 5.1590319, 2.4217949, 5.1596546, -2.3425913, 2.3455448

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639750, upper bound: 1.4669635
time: 4.48 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4655930, upper bound: 1.4669609
time: 4.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.3440685, -10.2844639, -14.3000622, -10.2871809, -2.7247262, 2.6730857
1: -12.5052509, -8.9332781, -12.4945755, -8.9361744, -2.6904726, 2.6779356
2: -13.4136314, -10.1668415, -13.4097490, -10.1796150, -2.6659775, 2.6778545
3: -9.8948479, -6.8911009, -9.8902311, -6.9025426, -2.7875118, 2.7941470
4: -4.5728383, -2.3956420, -4.5608406, -2.3998139, -1.6705756, 1.6626627
5: -11.0779305, -7.3603792, -11.0733852, -7.3661003, -2.6980753, 2.6911035
6: -17.6217384, -13.6018181, -17.5802155, -13.6031570, -3.0311108, 2.9813066
7: -6.4381933, -3.5917931, -6.4332142, -3.5954432, -2.2774324, 2.2758198
8: -2.0595164, 0.1869969, -2.0399003, 0.1837735, -1.8580909, 1.8410904
9: 2.4132347, 5.1638832, 2.4171619, 5.1602306, -2.3571682, 2.3553996

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4665448, upper bound: 1.4681542
time: 5.06 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4681542, upper bound: 1.4681544
time: 5.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.46 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 25.46
Output dim: 9, lower bound: -1.4639750, upper bound: 1.4669635
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 25.46
Output dim: 9, lower bound: -1.4655930, upper bound: 1.4669609
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 25.46
Output dim: 9, lower bound: -1.4665448, upper bound: 1.4681542
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 25.46
Output dim: 9, lower bound: -1.4681542, upper bound: 1.4681544

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206205, -14.2908955, -10.3032207, -2.6473169, 2.6388597
1: -12.4855118, -8.9540005, -12.4904757, -8.9445162, -2.6605339, 2.6583166
2: -13.3854198, -10.1927109, -13.3986130, -10.1854353, -2.6265864, 2.6353621
3: -9.8770170, -6.9295988, -9.8839674, -6.9145675, -2.7589808, 2.7529297
4: -4.5587468, -2.4179029, -4.5599356, -2.4083698, -1.6401732, 1.6336923
5: -11.0593300, -7.3853621, -11.0669880, -7.3745270, -2.6600161, 2.6533628
6: -17.5622883, -13.6298618, -17.5722466, -13.6159420, -2.9540796, 2.9507909
7: -6.4170847, -3.6060047, -6.4260955, -3.6002278, -2.2505960, 2.2519531
8: -2.0306225, 0.1645775, -2.0355787, 0.1748562, -1.8178439, 1.8107042
9: 2.4325776, 5.1581783, 2.4242964, 5.1592965, -2.3353109, 2.3408465

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639750, upper bound: 1.4653534
time: 4.72 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4639750, upper bound: 1.4669635
time: 4.93 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -14.3016462, -10.3013592, -14.2955542, -10.3028755, -2.6721640, 2.6724100
1: -12.4927616, -8.9171915, -12.4925337, -8.9427538, -2.6774530, 2.7060981
2: -13.3994026, -10.1215305, -13.4037924, -10.1824713, -2.6556253, 2.6972151
3: -9.9334059, -6.8938150, -9.8853674, -6.9041123, -2.8235502, 2.7890043
4: -4.5677190, -2.4049456, -4.5605354, -2.4070344, -1.6639657, 1.6490781
5: -11.1272087, -7.3633318, -11.0691519, -7.3670645, -2.7385788, 2.6798058
6: -17.6098251, -13.6224480, -17.5772438, -13.6154308, -3.0257797, 2.9927320
7: -6.4394741, -3.5618873, -6.4317551, -3.5980339, -2.2765889, 2.3066330
8: -2.0576124, 0.1731682, -2.0371175, 0.1777020, -1.8535433, 1.8355927
9: 2.4189563, 5.1642022, 2.4218001, 5.1596541, -2.3507488, 2.3512149

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4547419
time: 4.44 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4666765
time: 5.55 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -14.3332977, -10.2852621, -14.2954025, -10.2875309, -2.7127209, 2.6660748
1: -12.5004339, -8.9373798, -12.4925108, -8.9379435, -2.6833668, 2.6728334
2: -13.4016132, -10.1743374, -13.4045715, -10.1826468, -2.6451473, 2.6623468
3: -9.8916588, -6.9152098, -9.8888311, -6.9130077, -2.7756047, 2.7723494
4: -4.5713191, -2.3987157, -4.5602274, -2.4011550, -1.6612883, 1.6530421
5: -11.0728359, -7.3777108, -11.0712156, -7.3735809, -2.6829500, 2.6669841
6: -17.6096687, -13.6030006, -17.5752373, -13.6036682, -3.0150347, 2.9722443
7: -6.4250574, -3.5969315, -6.4275393, -3.5976453, -2.2650495, 2.2657340
8: -2.0559011, 0.1803875, -2.0383797, 0.1809244, -1.8492064, 1.8288212
9: 2.4190240, 5.1630201, 2.4196601, 5.1598730, -2.3498709, 2.3506761

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4559345
time: 5.56 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4678673
time: 4.49 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -14.3549156, -10.2660084, -14.3000565, -10.2871819, -2.7340937, 2.6996942
1: -12.5077085, -8.9004412, -12.4945669, -8.9361734, -2.7002664, 2.7207379
2: -13.4156017, -10.1035480, -13.4097452, -10.1796198, -2.6742239, 2.7189221
3: -9.9479713, -6.8794494, -9.8902283, -6.9025593, -2.8401036, 2.8080912
4: -4.5802431, -2.3856273, -4.5608406, -2.3998165, -1.6804280, 1.6683915
5: -11.1407995, -7.3556747, -11.0733814, -7.3661175, -2.7573867, 2.6935391
6: -17.6569729, -13.5955944, -17.5802116, -13.6031599, -3.0663676, 3.0140047
7: -6.4474144, -3.5527616, -6.4331951, -3.5954480, -2.2910757, 2.3207433
8: -2.0828876, 0.1889720, -2.0398984, 0.1837654, -1.8756862, 1.8536391
9: 2.4053869, 5.1690493, 2.4171648, 5.1602302, -2.3652554, 2.3610501

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678659, upper bound: 1.4559345
time: 4.71 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678659, upper bound: 1.4678658
time: 5.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.28 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4639750, upper bound: 1.4653534
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4639750, upper bound: 1.4669635
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4547419
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4666765
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4559345
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4678673
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4678659, upper bound: 1.4559345
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4678659, upper bound: 1.4678658

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206205, -14.2848425, -10.3036900, -2.6459274, 2.6325560
1: -12.4855118, -8.9540005, -12.4877596, -8.9468918, -2.6588554, 2.6549926
2: -13.3854198, -10.1927109, -13.3917723, -10.1897449, -2.6223116, 2.6251969
3: -9.8770170, -6.9295988, -9.8822441, -6.9278784, -2.7479515, 2.7513013
4: -4.5587468, -2.4179029, -4.5590501, -2.4101617, -1.6362438, 1.6301408
5: -11.0593300, -7.3853621, -11.0638494, -7.3843260, -2.6476631, 2.6501422
6: -17.5622883, -13.6298618, -17.5654335, -13.6166105, -2.9521575, 2.9430122
7: -6.4170847, -3.6060047, -6.4186120, -3.6032424, -2.2470918, 2.2462707
8: -2.0306225, 0.1645775, -2.0335951, 0.1710916, -1.8117542, 1.8082533
9: 2.4325776, 5.1581783, 2.4275889, 5.1587973, -2.3342350, 2.3371711

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4531330
time: 4.83 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4650659
time: 4.88 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206205, -14.3065643, -10.2844200, -2.6748838, 2.6537161
1: -12.4855118, -8.9540005, -12.4949942, -8.9099426, -2.7042503, 2.6649318
2: -13.3854198, -10.1927109, -13.4057512, -10.1184340, -2.6776199, 2.6401496
3: -9.8770170, -6.9295988, -9.9386520, -6.8922925, -2.7857046, 2.8060079
4: -4.5587468, -2.4179029, -4.5680552, -2.3971133, -1.6490602, 1.6399286
5: -11.0593300, -7.3853621, -11.1318817, -7.3622952, -2.6694598, 2.7188146
6: -17.5622883, -13.6298618, -17.6131897, -13.6092014, -2.9881659, 3.0085645
7: -6.4170847, -3.6060047, -6.4409723, -3.5590286, -2.2975426, 2.2694507
8: -2.0306225, 0.1645775, -2.0606651, 0.1796780, -1.8214550, 1.8413184
9: 2.4325776, 5.1581783, 2.4139738, 5.1648369, -2.3418550, 2.3509474

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4547424
time: 5.40 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4666751
time: 5.12 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -14.2969637, -10.3056793, -14.2937603, -10.3034372, -2.6669989, 2.6668935
1: -12.4765329, -8.9240980, -12.4854546, -8.9440136, -2.6602864, 2.6924410
2: -13.3950863, -10.1241970, -13.4026756, -10.1830349, -2.6483488, 2.6922476
3: -9.9291859, -6.8954368, -9.8839178, -6.9044647, -2.8169551, 2.7813401
4: -4.5532007, -2.4129722, -4.5541854, -2.4083879, -1.6484914, 1.6355059
5: -11.1234884, -7.3681226, -11.0682182, -7.3686323, -2.7314882, 2.6741142
6: -17.5997391, -13.6285324, -17.5730858, -13.6166906, -3.0138164, 2.9825516
7: -6.4351735, -3.5703773, -6.4309459, -3.6014414, -2.2668114, 2.2970221
8: -2.0523562, 0.1680994, -2.0360770, 0.1754637, -1.8443136, 1.8291113
9: 2.4301715, 5.1506891, 2.4241595, 5.1536021, -2.3334329, 2.3352699

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4535879
time: 5.40 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4547419
time: 5.23 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -14.3016415, -10.3013592, -14.2955513, -10.3028746, -2.6719651, 2.6721191
1: -12.4927559, -8.9171963, -12.4925327, -8.9427547, -2.6717930, 2.7060966
2: -13.3994026, -10.1215343, -13.4037914, -10.1824703, -2.6550398, 2.6953783
3: -9.9334030, -6.8938179, -9.8853683, -6.9041119, -2.8217745, 2.7945542
4: -4.5677137, -2.4049475, -4.5605350, -2.4070330, -1.6522315, 1.6490757
5: -11.1272068, -7.3633337, -11.0691528, -7.3670635, -2.7387962, 2.6797085
6: -17.6098232, -13.6224499, -17.5772438, -13.6154318, -3.0215168, 2.9927311
7: -6.4394732, -3.5618916, -6.4317551, -3.5980325, -2.2762465, 2.3026805
8: -2.0576110, 0.1731668, -2.0371170, 0.1777010, -1.8521938, 1.8324397
9: 2.4189587, 5.1641994, 2.4218006, 5.1596537, -2.3507450, 2.3454595

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4655200
time: 4.65 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4666765
time: 4.72 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -14.3285809, -10.2895708, -14.2936096, -10.2880974, -2.7073927, 2.6605697
1: -12.4841938, -8.9443293, -12.4854288, -8.9392061, -2.6661868, 2.6591172
2: -13.3973303, -10.1770477, -13.4034576, -10.1832104, -2.6378512, 2.6573424
3: -9.8874311, -6.9168262, -9.8873806, -6.9133582, -2.7707162, 2.7646832
4: -4.5567980, -2.4067566, -4.5538769, -2.4025111, -1.6458144, 1.6394880
5: -11.0691080, -7.3824968, -11.0702801, -7.3751507, -2.6758204, 2.6612935
6: -17.5995598, -13.6090937, -17.5710735, -13.6049271, -3.0030093, 2.9620614
7: -6.4207535, -3.6054564, -6.4267273, -3.6010613, -2.2552762, 2.2560189
8: -2.0506783, 0.1753235, -2.0373411, 0.1786880, -1.8414130, 1.8223467
9: 2.4303093, 5.1495018, 2.4220243, 5.1538210, -2.3325946, 2.3347239

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4543270
time: 4.98 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4559345
time: 4.83 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -14.3332949, -10.2852621, -14.2953987, -10.2875309, -2.7124987, 2.6657825
1: -12.5004272, -8.9373798, -12.4925079, -8.9379425, -2.6777077, 2.6728311
2: -13.4016123, -10.1743402, -13.4045706, -10.1826448, -2.6445642, 2.6610374
3: -9.8916559, -6.9152098, -9.8888321, -6.9130077, -2.7753172, 2.7778983
4: -4.5713158, -2.3987179, -4.5602279, -2.4011564, -1.6495538, 1.6530402
5: -11.0728350, -7.3777146, -11.0712137, -7.3735800, -2.6848226, 2.6668868
6: -17.6096668, -13.6030006, -17.5752373, -13.6036682, -3.0110517, 2.9722438
7: -6.4250565, -3.5969357, -6.4275422, -3.5976486, -2.2647076, 2.2577641
8: -2.0558996, 0.1803856, -2.0383787, 0.1809225, -1.8492045, 1.8256664
9: 2.4190264, 5.1630154, 2.4196601, 5.1598730, -2.3498678, 2.3449199

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4662611
time: 4.60 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4678674
time: 4.54 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -14.3501587, -10.2703190, -14.2982674, -10.2877474, -2.7287569, 2.6941886
1: -12.4914598, -8.9074078, -12.4874878, -8.9374371, -2.6830893, 2.7070303
2: -13.4113274, -10.1062279, -13.4086304, -10.1801834, -2.6669273, 2.7139444
3: -9.9437532, -6.8810663, -9.8887787, -6.9029102, -2.8335009, 2.8004270
4: -4.5657234, -2.3936605, -4.5544896, -2.4011705, -1.6649160, 1.6548378
5: -11.1370764, -7.3604593, -11.0724506, -7.3676872, -2.7502990, 2.6878543
6: -17.6468582, -13.6016884, -17.5760460, -13.6044140, -3.0543871, 3.0038180
7: -6.4431086, -3.5612941, -6.4323854, -3.5988626, -2.2813001, 2.3110826
8: -2.0776596, 0.1839032, -2.0388632, 0.1815286, -1.8664598, 1.8471632
9: 2.4166737, 5.1555319, 2.4195309, 5.1541786, -2.3479834, 2.3450997

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4674085, upper bound: 1.4536304
time: 4.75 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678607, upper bound: 1.4559291
time: 4.93 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -14.3549118, -10.2660093, -14.3000546, -10.2871799, -2.7338672, 2.6994061
1: -12.5077019, -8.9004421, -12.4945669, -8.9361744, -2.6946073, 2.7207346
2: -13.4156017, -10.1035509, -13.4097462, -10.1796179, -2.6736403, 2.7170856
3: -9.9479713, -6.8794489, -9.8902273, -6.9025593, -2.8383269, 2.8136392
4: -4.5802383, -2.3856285, -4.5608392, -2.3998182, -1.6685386, 1.6683908
5: -11.1407967, -7.3556747, -11.0733833, -7.3661194, -2.7576103, 2.6934433
6: -17.6569691, -13.5955963, -17.5802078, -13.6031590, -3.0621042, 3.0140018
7: -6.4474111, -3.5527673, -6.4331970, -3.5954485, -2.2907324, 2.3167744
8: -2.0828876, 0.1889706, -2.0398998, 0.1837645, -1.8743410, 1.8504856
9: 2.4053893, 5.1690454, 2.4171653, 5.1602283, -2.3652530, 2.3552954

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4674085, upper bound: 1.4655607
time: 4.44 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678607, upper bound: 1.4678620
time: 5.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.87 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4531330
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4650659
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4547424
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4666751
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4535879
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4547419
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4655200
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4653073, upper bound: 1.4666765
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4543270
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4559345
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4662611
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4678674
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4674085, upper bound: 1.4536304
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4678607, upper bound: 1.4559291
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4674085, upper bound: 1.4655607
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.87
Output dim: 9, lower bound: -1.4678607, upper bound: 1.4678620

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.2830572, -10.3042564, -2.6407557, 2.6270394
1: -12.4692907, -8.9608955, -12.4806786, -8.9481602, -2.6416874, 2.6413236
2: -13.3810949, -10.1954117, -13.3906574, -10.1903114, -2.6150351, 2.6201982
3: -9.8727856, -6.9312177, -9.8807936, -6.9282312, -2.7430644, 2.7436323
4: -4.5442271, -2.4259353, -4.5526981, -2.4115164, -1.6207690, 1.6165669
5: -11.0555973, -7.3901548, -11.0629158, -7.3858967, -2.6405163, 2.6444454
6: -17.5522270, -13.6359434, -17.5612755, -13.6178713, -2.9401803, 2.9328313
7: -6.4127755, -3.6144905, -6.4178019, -3.6066527, -2.2373209, 2.2365971
8: -2.0253696, 0.1595144, -2.0325546, 0.1688547, -1.8039308, 1.8017738
9: 2.4437881, 5.1446662, 2.4299502, 5.1527457, -2.3169122, 2.3212237

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636945, upper bound: 1.4519741
time: 5.44 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636945, upper bound: 1.4531331
time: 5.15 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206253, -14.2848434, -10.3036890, -2.6457291, 2.6322646
1: -12.4855061, -8.9540024, -12.4877558, -8.9468927, -2.6531935, 2.6549902
2: -13.3854179, -10.1927128, -13.3917713, -10.1897459, -2.6217241, 2.6238890
3: -9.8770123, -6.9295988, -9.8822441, -6.9278774, -2.7476664, 2.7568483
4: -4.5587440, -2.4179046, -4.5590477, -2.4101624, -1.6245093, 1.6301389
5: -11.0593290, -7.3853664, -11.0638485, -7.3843269, -2.6495337, 2.6500449
6: -17.5622864, -13.6298618, -17.5654335, -13.6166134, -2.9481745, 2.9430094
7: -6.4170828, -3.6060100, -6.4186125, -3.6032438, -2.2467537, 2.2383010
8: -2.0306215, 0.1645761, -2.0335951, 0.1710906, -1.8117528, 1.8051002
9: 2.4325800, 5.1581740, 2.4275904, 5.1587963, -2.3342321, 2.3314145

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636922, upper bound: 1.4639050
time: 4.80 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636922, upper bound: 1.4650659
time: 4.82 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.3047705, -10.2849884, -2.6697116, 2.6481957
1: -12.4692907, -8.9608955, -12.4879131, -8.9112110, -2.6870866, 2.6512637
2: -13.3810949, -10.1954117, -13.4046383, -10.1189804, -2.6703739, 2.6351538
3: -9.8727856, -6.9312177, -9.9372025, -6.8926458, -2.7808166, 2.8002496
4: -4.5442271, -2.4259353, -4.5617056, -2.3984652, -1.6335859, 1.6263561
5: -11.0555973, -7.3901548, -11.1309471, -7.3638654, -2.6623187, 2.7131419
6: -17.5522270, -13.6359434, -17.6090279, -13.6104612, -2.9761887, 2.9983869
7: -6.4127755, -3.6144905, -6.4401655, -3.5624409, -2.2875352, 2.2597737
8: -2.0253696, 0.1595144, -2.0596266, 0.1774387, -1.8136292, 1.8348186
9: 2.4437881, 5.1446662, 2.4163356, 5.1587849, -2.3245330, 2.3350000

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4535882
time: 5.38 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4547424
time: 5.38 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206253, -14.3065634, -10.2844181, -2.6746864, 2.6534257
1: -12.4855061, -8.9540024, -12.4949932, -8.9099426, -2.6985884, 2.6649284
2: -13.3854179, -10.1927128, -13.4057522, -10.1184320, -2.6790733, 2.6388426
3: -9.8770123, -6.9295988, -9.9386520, -6.8922915, -2.7854176, 2.8081493
4: -4.5587440, -2.4179046, -4.5680532, -2.3971131, -1.6373258, 1.6399276
5: -11.0593290, -7.3853664, -11.1318779, -7.3622975, -2.6713328, 2.7183857
6: -17.5622864, -13.6298618, -17.6131897, -13.6092014, -2.9841828, 3.0085621
7: -6.4170828, -3.6060100, -6.4409723, -3.5590291, -2.2954016, 2.2614803
8: -2.0306215, 0.1645761, -2.0606651, 0.1796770, -1.8214531, 1.8380361
9: 2.4325800, 5.1581740, 2.4139743, 5.1648369, -2.3418527, 2.3451922

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636937, upper bound: 1.4655209
time: 4.90 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4636937, upper bound: 1.4666758
time: 5.04 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2969637, -10.3056793, -14.2888298, -10.3203745, -2.6497388, 2.6629357
1: -12.4765329, -8.9240980, -12.4832077, -8.9510612, -2.6523323, 2.6884832
2: -13.3950863, -10.1241970, -13.3963337, -10.1861610, -2.6436739, 2.6859336
3: -9.9291859, -6.8954368, -9.8787041, -6.9061775, -2.8147912, 2.7757554
4: -4.5532007, -2.4129722, -4.5538526, -2.4161401, -1.6393530, 1.6324615
5: -11.1234884, -7.3681226, -11.0636911, -7.3696680, -2.7275305, 2.6670671
6: -17.5997391, -13.6285324, -17.5698643, -13.6299314, -2.9994235, 2.9773879
7: -6.4351735, -3.5703773, -6.4294167, -3.6042018, -2.2629962, 2.2943153
8: -2.0523562, 0.1680994, -2.0330281, 0.1689525, -1.8389735, 1.8254657
9: 2.4301715, 5.1506891, 2.4291325, 5.1529799, -2.3308351, 2.3297110

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4535895
time: 4.85 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4535880
time: 4.80 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2969637, -10.3056793, -14.3422518, -10.2850266, -2.6859350, 2.6915343
1: -12.4765329, -8.9240980, -12.4981613, -8.9345531, -2.6692882, 2.7041364
2: -13.3950863, -10.1241970, -13.4125042, -10.1674156, -2.6639457, 2.6984987
3: -9.9291859, -6.8954368, -9.8933935, -6.8914690, -2.8225994, 2.7906919
4: -4.5532007, -2.4129722, -4.5664849, -2.3970041, -1.6576662, 1.6452231
5: -11.1234884, -7.3681226, -11.0769939, -7.3619576, -2.7326422, 2.6810021
6: -17.5997391, -13.6285324, -17.6175613, -13.6030817, -3.0174623, 3.0124750
7: -6.4351735, -3.5703773, -6.4373655, -3.5952153, -2.2740040, 2.3043675
8: -2.0523562, 0.1680994, -2.0584826, 0.1847534, -1.8489285, 1.8453538
9: 2.4301715, 5.1506891, 2.4156251, 5.1578288, -2.3355603, 2.3430109

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547427
time: 4.86 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547448
time: 4.82 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3016415, -10.3013592, -14.2906151, -10.3198051, -2.6547070, 2.6681638
1: -12.4927559, -8.9171963, -12.4902859, -8.9498043, -2.6638422, 2.7021384
2: -13.3994026, -10.1215343, -13.3974457, -10.1855955, -2.6503658, 2.6890619
3: -9.9334030, -6.8938179, -9.8801537, -6.9058237, -2.8196120, 2.7889709
4: -4.5677137, -2.4049475, -4.5602040, -2.4147861, -1.6430964, 1.6460319
5: -11.1272068, -7.3633337, -11.0646267, -7.3681002, -2.7352715, 2.6726604
6: -17.6098232, -13.6224499, -17.5740242, -13.6286736, -3.0074210, 2.9875751
7: -6.4394732, -3.5618916, -6.4302273, -3.6007986, -2.2724323, 2.2999737
8: -2.0576110, 0.1731668, -2.0340748, 0.1711893, -1.8468552, 1.8287997
9: 2.4189587, 5.1641994, 2.4267764, 5.1590309, -2.3481479, 2.3399081

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533775, upper bound: 1.4655223
time: 5.36 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533752, upper bound: 1.4655211
time: 5.99 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3016415, -10.3013592, -14.3440628, -10.2844667, -2.6908975, 2.6965737
1: -12.4927559, -8.9171963, -12.5052452, -8.9332800, -2.6808214, 2.7177978
2: -13.3994026, -10.1215343, -13.4136257, -10.1668453, -2.6706390, 2.7016354
3: -9.9334030, -6.8938179, -9.8948441, -6.8911176, -2.8274188, 2.8039069
4: -4.5677137, -2.4049475, -4.5728364, -2.3956447, -1.6613002, 1.6587946
5: -11.1272068, -7.3633337, -11.0779285, -7.3603926, -2.7399411, 2.6866007
6: -17.6098232, -13.6224499, -17.6217346, -13.6018181, -3.0251656, 3.0226173
7: -6.4394732, -3.5618916, -6.4381752, -3.5917959, -2.2834496, 2.3100407
8: -2.0576110, 0.1731668, -2.0595140, 0.1869898, -1.8568115, 1.8485768
9: 2.4189587, 5.1641994, 2.4132385, 5.1638818, -2.3528762, 2.3532419

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533752, upper bound: 1.4666759
time: 5.54 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533752, upper bound: 1.4666756
time: 5.66 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -14.3285809, -10.2895708, -14.2875652, -10.2885590, -2.7060022, 2.6542230
1: -12.4841938, -8.9443293, -12.4827118, -8.9415302, -2.6644373, 2.6557932
2: -13.3973303, -10.1770477, -13.3966236, -10.1876106, -2.6336942, 2.6471772
3: -9.8874311, -6.9168262, -9.8856621, -6.9266710, -2.7596889, 2.7630577
4: -4.5567980, -2.4067566, -4.5529733, -2.4043078, -1.6418908, 1.6359358
5: -11.0691080, -7.3824968, -11.0671406, -7.3849478, -2.6634684, 2.6580029
6: -17.5995598, -13.6090937, -17.5641670, -13.6055984, -3.0011044, 2.9543872
7: -6.4207535, -3.6054564, -6.4192476, -3.6040769, -2.2517929, 2.2503383
8: -2.0506783, 0.1753235, -2.0353785, 0.1749277, -1.8353252, 1.8199687
9: 2.4303093, 5.1495018, 2.4253130, 5.1533208, -2.3315272, 2.3310404

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4658006, upper bound: 1.4520213
time: 5.12 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662539, upper bound: 1.4543217
time: 4.89 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -14.3285809, -10.2895708, -14.3092508, -10.2692871, -2.7142215, 2.6753612
1: -12.4841938, -8.9443293, -12.4899311, -8.9044571, -2.7098575, 2.6657944
2: -13.3973303, -10.1770477, -13.4106026, -10.1161499, -2.6892123, 2.6621041
3: -9.8874311, -6.9168262, -9.9420929, -6.8912587, -2.7971601, 2.8139832
4: -4.5567980, -2.4067566, -4.5620141, -2.3911743, -1.6547687, 1.6457458
5: -11.0691080, -7.3824968, -11.1353092, -7.3629179, -2.6852574, 2.7250526
6: -17.5995598, -13.6090937, -17.6121254, -13.5981874, -3.0265832, 3.0201674
7: -6.4207535, -3.6054564, -6.4415865, -3.5597796, -2.3021879, 2.2734308
8: -2.0506783, 0.1753235, -2.0626726, 0.1835117, -1.8449869, 1.8530734
9: 2.4303093, 5.1495018, 2.4116936, 5.1593933, -2.3391659, 2.3447924

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4658006, upper bound: 1.4536304
time: 4.55 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662539, upper bound: 1.4559315
time: 4.95 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -14.3332949, -10.2852621, -14.2893496, -10.2879944, -2.7111092, 2.6594367
1: -12.5004272, -8.9373798, -12.4897881, -8.9402599, -2.6759586, 2.6695070
2: -13.4016123, -10.1743402, -13.3977385, -10.1870441, -2.6404042, 2.6508722
3: -9.8916559, -6.9152098, -9.8871098, -6.9263182, -2.7642899, 2.7762733
4: -4.5713158, -2.3987179, -4.5593252, -2.4029536, -1.6456301, 1.6494889
5: -11.0728350, -7.3777146, -11.0680742, -7.3833771, -2.6724701, 2.6635971
6: -17.6096668, -13.6030006, -17.5683270, -13.6043386, -3.0091534, 2.9645681
7: -6.4250565, -3.5969357, -6.4200573, -3.6006618, -2.2612262, 2.2520833
8: -2.0558996, 0.1803856, -2.0364161, 0.1771617, -1.8431182, 1.8232880
9: 2.4190264, 5.1630154, 2.4229498, 5.1593719, -2.3488007, 2.3412378

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4658006, upper bound: 1.4639531
time: 5.21 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662539, upper bound: 1.4662557
time: 4.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.20 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636945, upper bound: 1.4519741
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636945, upper bound: 1.4531331
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636922, upper bound: 1.4639050
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636922, upper bound: 1.4650659
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4535882
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636913, upper bound: 1.4547424
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636937, upper bound: 1.4655209
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4636937, upper bound: 1.4666758
IS_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4535895
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4535880
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547427
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4629705, upper bound: 1.4547448
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4533775, upper bound: 1.4655223
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4533752, upper bound: 1.4655211
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4533752, upper bound: 1.4666759
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4533752, upper bound: 1.4666756
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4658006, upper bound: 1.4520213
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4662539, upper bound: 1.4543217
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4658006, upper bound: 1.4536304
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4662539, upper bound: 1.4559315
IS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4658006, upper bound: 1.4639531
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 31.20
Output dim: 9, lower bound: -1.4662539, upper bound: 1.4662557
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 9, lower bound: -1.4662591, upper bound: 1.4678674
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 9, lower bound: -1.4674085, upper bound: 1.4536304
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 9, lower bound: -1.4678607, upper bound: 1.4559291
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 9, lower bound: -1.4674085, upper bound: 1.4655607
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 9, lower bound: -1.4678607, upper bound: 1.4678620
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3557186126708984
rel_dist={9: [-1.46816030629895, 1.4681598898989336]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5816
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5816

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3577571, upper bound: 1.3586908
time: 5.47 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600242, upper bound: 1.3600239
time: 4.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.68
Output dim: 9, lower bound: -1.3577571, upper bound: 1.3586908
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.68
Output dim: 9, lower bound: -1.3600242, upper bound: 1.3600239

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2906237, -10.3198071, -14.2945557, -10.3063259, -2.5062790, 2.4956923
1: -12.4902925, -8.9498024, -12.4920864, -8.9441948, -2.5464087, 2.5432220
2: -13.3974476, -10.1855946, -13.4025002, -10.1830988, -2.5270224, 2.5293975
3: -9.8801556, -6.9058075, -9.8843050, -6.9044418, -2.7008696, 2.7035332
4: -4.5602050, -2.4147825, -4.5604696, -2.4086161, -1.5759730, 1.5711219
5: -11.0646286, -7.3680825, -11.0682297, -7.3672585, -2.5426579, 2.5447459
6: -17.5740318, -13.6286716, -17.5765991, -13.6181297, -2.8385420, 2.8310452
7: -6.4302454, -3.6007943, -6.4314623, -3.5985956, -2.1816587, 2.1809652
8: -2.0340772, 0.1711969, -2.0365028, 0.1763811, -1.7584825, 1.7557676
9: 2.4267726, 5.1590319, 2.4228139, 5.1595292, -2.2862992, 2.2886527

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3563727, upper bound: 1.3586865
time: 5.77 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3577559, upper bound: 1.3586874
time: 7.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.3440685, -10.2844639, -14.3000612, -10.2871876, -2.5727825, 2.5217552
1: -12.5052509, -8.9332781, -12.4945755, -8.9361744, -2.5710220, 2.5575604
2: -13.4136314, -10.1668415, -13.4097509, -10.1796150, -2.5458717, 2.5583830
3: -9.8948479, -6.8911009, -9.8902283, -6.9025426, -2.7179184, 2.7245979
4: -4.5728383, -2.3956420, -4.5608406, -2.3998160, -1.5990362, 1.5898967
5: -11.0779305, -7.3603792, -11.0733852, -7.3661017, -2.5661573, 2.5599165
6: -17.6217384, -13.6018181, -17.5802135, -13.6031570, -2.9031148, 2.8512769
7: -6.4381933, -3.5917931, -6.4332132, -3.5954461, -2.1967564, 2.1951146
8: -2.0595164, 0.1869969, -2.0399008, 0.1837711, -1.7914863, 1.7740788
9: 2.4132347, 5.1638832, 2.4171629, 5.1602311, -2.3009441, 2.2996385

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6222
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6222

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586172, upper bound: 1.3600226
time: 5.37 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3600222
time: 5.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.92 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 24.92
Output dim: 9, lower bound: -1.3563727, upper bound: 1.3586865
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 24.92
Output dim: 9, lower bound: -1.3577559, upper bound: 1.3586874
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 24.92
Output dim: 9, lower bound: -1.3586172, upper bound: 1.3600226
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.92
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3600222

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206205, -14.2886181, -10.3067732, -2.4940782, 2.4873161
1: -12.4855118, -8.9540005, -12.4894552, -8.9464588, -2.5391369, 2.5373755
2: -13.3854198, -10.1927109, -13.3958874, -10.1869116, -2.5052834, 2.5122118
3: -9.8770170, -6.9295988, -9.8825274, -6.9177465, -2.6865969, 2.6818933
4: -4.5587468, -2.4179029, -4.5596948, -2.4103255, -1.5659490, 1.5607996
5: -11.0593300, -7.3853621, -11.0654354, -7.3767905, -2.5253415, 2.5201092
6: -17.5622883, -13.6298618, -17.5702133, -13.6187840, -2.8227215, 2.8200312
7: -6.4170847, -3.6060047, -6.4242215, -3.6014142, -2.1684194, 2.1694918
8: -2.0306225, 0.1645775, -2.0345554, 0.1727414, -1.7485414, 1.7428694
9: 2.4325776, 5.1581783, 2.4260044, 5.1590700, -2.2787907, 2.2831891

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3496369
time: 4.72 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3584042
time: 5.02 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -14.3016462, -10.3013592, -14.2945490, -10.3063259, -2.5185757, 2.5221677
1: -12.4927616, -8.9171915, -12.4920769, -8.9441938, -2.5560827, 2.5858369
2: -13.3994026, -10.1215305, -13.4024925, -10.1831036, -2.5333633, 2.5748301
3: -9.9334059, -6.8938150, -9.8843031, -6.9044614, -2.7510157, 2.7170167
4: -4.5677190, -2.4049456, -4.5604701, -2.4086175, -1.5899925, 1.5769193
5: -11.1272087, -7.3633318, -11.0682259, -7.3672757, -2.6039181, 2.5448713
6: -17.6098251, -13.6224480, -17.5765915, -13.6181335, -2.8922491, 2.8636699
7: -6.4394741, -3.5618873, -6.4314394, -3.5985994, -2.1944609, 2.2241187
8: -2.0576124, 0.1731682, -2.0365009, 0.1763725, -1.7841787, 1.7673180
9: 2.4189563, 5.1642022, 2.4228182, 5.1595278, -2.2940814, 2.2943227

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3496336
time: 5.44 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3584071
time: 4.79 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -14.3332977, -10.2852621, -14.2941256, -10.2876320, -2.5604978, 2.5134258
1: -12.5004339, -8.9373798, -12.4919395, -8.9384499, -2.5635581, 2.5517769
2: -13.4016132, -10.1743374, -13.4031439, -10.1835270, -2.5241623, 2.5407605
3: -9.8916588, -6.9152098, -9.8884554, -6.9158382, -2.7036285, 2.7024522
4: -4.5713191, -2.3987157, -4.5600491, -2.4015317, -1.5889254, 1.5795398
5: -11.0728359, -7.3777108, -11.0705862, -7.3756351, -2.5484462, 2.5351305
6: -17.6096687, -13.6030006, -17.5738373, -13.6038094, -2.8866491, 2.8406439
7: -6.4250574, -3.5969315, -6.4259768, -3.5982680, -2.1836538, 2.1838605
8: -2.0559011, 0.1803875, -2.0379572, 0.1801376, -1.7813416, 1.7613201
9: 2.4190240, 5.1630201, 2.4203506, 5.1597695, -2.2934208, 2.2941449

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583370, upper bound: 1.3509721
time: 5.46 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3597421
time: 4.63 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -14.3549156, -10.2660084, -14.3000526, -10.2871838, -2.5815196, 2.5483642
1: -12.5077085, -8.9004412, -12.4945660, -8.9361782, -2.5805130, 2.6003613
2: -13.4156017, -10.1035480, -13.4097443, -10.1796179, -2.5522814, 2.5979848
3: -9.9479713, -6.8794494, -9.8902264, -6.9025612, -2.7679591, 2.7372403
4: -4.5802431, -2.3856273, -4.5608401, -2.3998194, -1.6071908, 1.5956240
5: -11.1407995, -7.3556747, -11.0733824, -7.3661222, -2.6228080, 2.5600386
6: -17.6569729, -13.5955944, -17.5802078, -13.6031609, -2.9355068, 2.8839746
7: -6.4474144, -3.5527616, -6.4331918, -3.5954485, -2.2097268, 2.2387786
8: -2.0828876, 0.1889720, -2.0398965, 0.1837621, -1.8076153, 1.7856929
9: 2.4053869, 5.1690493, 2.4171677, 5.1602302, -2.3086576, 2.3052886

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 833

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597429, upper bound: 1.3509703
time: 5.30 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3597420
time: 4.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.69 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3496369
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3584042
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3496336
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3584071
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3583370, upper bound: 1.3509721
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3597421
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3597429, upper bound: 1.3509703
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.69
Output dim: 9, lower bound: -1.3597406, upper bound: 1.3597420

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.2865295, -10.3074541, -2.4887409, 2.4815679
1: -12.4692907, -8.9608955, -12.4809895, -8.9479704, -2.5217657, 2.5223465
2: -13.3810949, -10.1954117, -13.3945541, -10.1875896, -2.4978600, 2.5069942
3: -9.8727856, -6.9312177, -9.8807936, -6.9181666, -2.6816254, 2.6735249
4: -4.5442271, -2.4259353, -4.5521002, -2.4119475, -1.5502548, 1.5460696
5: -11.0555973, -7.3901548, -11.0643177, -7.3786683, -2.5176973, 2.5142355
6: -17.5522270, -13.6359434, -17.5652390, -13.6202898, -2.8105063, 2.8090639
7: -6.4127755, -3.6144905, -6.4232502, -3.6054893, -2.1577115, 2.1596382
8: -2.0253696, 0.1595144, -2.0333056, 0.1700664, -1.7402606, 1.7361541
9: 2.4437881, 5.1446662, 2.4288282, 5.1518321, -2.2602758, 2.2667706

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3482352
time: 4.58 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3496369
time: 4.47 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206253, -14.2886181, -10.3067741, -2.4937363, 2.4869580
1: -12.4855061, -8.9540024, -12.4894543, -8.9464607, -2.5331917, 2.5373721
2: -13.3854179, -10.1927128, -13.3958864, -10.1869116, -2.5042677, 2.5106668
3: -9.8770123, -6.9295988, -9.8825274, -6.9177461, -2.6860838, 2.6871953
4: -4.5587440, -2.4179046, -4.5596943, -2.4103260, -1.5536268, 1.5607967
5: -11.0593290, -7.3853664, -11.0654345, -7.3767910, -2.5271301, 2.5199356
6: -17.5622864, -13.6298618, -17.5702095, -13.6187859, -2.8185415, 2.8200288
7: -6.4170828, -3.6060100, -6.4242215, -3.6014152, -2.1680794, 2.1610723
8: -2.0306215, 0.1645761, -2.0345540, 0.1727409, -1.7485390, 1.7395585
9: 2.4325800, 5.1581740, 2.4260054, 5.1590676, -2.2787871, 2.2771449

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3570050
time: 5.07 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3584046
time: 5.07 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -14.2969637, -10.3056793, -14.2924538, -10.3070059, -2.5132446, 2.5164185
1: -12.4765329, -8.9240980, -12.4836140, -8.9457054, -2.5387125, 2.5708184
2: -13.3950863, -10.1241970, -13.4011593, -10.1837797, -2.5259390, 2.5696495
3: -9.9291859, -6.8954368, -9.8825712, -6.9048834, -2.7443395, 2.7086511
4: -4.5532007, -2.4129722, -4.5528741, -2.4102397, -1.5743003, 1.5621908
5: -11.1234884, -7.3681226, -11.0671110, -7.3691502, -2.5966353, 2.5390015
6: -17.5997391, -13.6285324, -17.5716171, -13.6196394, -2.8800302, 2.8527026
7: -6.4351735, -3.5703773, -6.4304714, -3.6026735, -2.1837449, 2.2143288
8: -2.0523562, 0.1680994, -2.0352535, 0.1736965, -1.7746334, 1.7606006
9: 2.4301715, 5.1506891, 2.4256434, 5.1522903, -2.2755749, 2.2779045

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3487952
time: 6.08 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3496336
time: 5.64 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -14.3016415, -10.3013592, -14.2945499, -10.3063269, -2.5182328, 2.5218182
1: -12.4927559, -8.9171963, -12.4920769, -8.9441948, -2.5501385, 2.5858340
2: -13.3994026, -10.1215343, -13.4024944, -10.1831036, -2.5323467, 2.5726871
3: -9.9334030, -6.8938179, -9.8843031, -6.9044619, -2.7489243, 2.7223196
4: -4.5677137, -2.4049475, -4.5604682, -2.4086189, -1.5776711, 1.5769167
5: -11.1272068, -7.3633337, -11.0682278, -7.3672767, -2.6040521, 2.5446963
6: -17.6098232, -13.6224499, -17.5765915, -13.6181335, -2.8875871, 2.8636680
7: -6.4394732, -3.5618916, -6.4314399, -3.5985990, -2.1941185, 2.2197690
8: -2.0576110, 0.1731668, -2.0364990, 0.1763716, -1.7828293, 1.7640071
9: 2.4189587, 5.1641994, 2.4228196, 5.1595268, -2.2940767, 2.2882788

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3575662
time: 5.72 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3584071
time: 5.14 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -14.3285809, -10.2895708, -14.2919893, -10.2883110, -2.5550051, 2.5076895
1: -12.4841938, -8.9443293, -12.4834766, -8.9399595, -2.5461788, 2.5367012
2: -13.3973303, -10.1770477, -13.4018097, -10.1842022, -2.5167170, 2.5355363
3: -9.8874311, -6.9168262, -9.8867197, -6.9162593, -2.6986551, 2.6940842
4: -4.5567980, -2.4067566, -4.5524530, -2.4031513, -1.5732334, 1.5648291
5: -11.0691080, -7.3824968, -11.0694685, -7.3775120, -2.5408177, 2.5292640
6: -17.5995598, -13.6090937, -17.5688591, -13.6053181, -2.8743782, 2.8296766
7: -6.4207535, -3.6054564, -6.4250059, -3.6023512, -2.1729431, 2.1739655
8: -2.0506783, 0.1753235, -2.0367136, 0.1774631, -1.7730918, 1.7546122
9: 2.4303093, 5.1495018, 2.4231801, 5.1525326, -2.2749538, 2.2777262

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3495653
time: 4.83 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583370, upper bound: 1.3509721
time: 5.05 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -14.3332949, -10.2852621, -14.2941227, -10.2876320, -2.5601592, 2.5131297
1: -12.5004272, -8.9373798, -12.4919386, -8.9384489, -2.5576162, 2.5517731
2: -13.4016123, -10.1743402, -13.4031429, -10.1835270, -2.5231457, 2.5392118
3: -9.8916559, -6.9152098, -9.8884544, -6.9158378, -2.7031150, 2.7077541
4: -4.5713158, -2.3987179, -4.5600481, -2.4015326, -1.5766029, 1.5795374
5: -11.0728350, -7.3777146, -11.0705853, -7.3756342, -2.5502348, 2.5349569
6: -17.6096668, -13.6030006, -17.5738335, -13.6038122, -2.8824654, 2.8406420
7: -6.4250565, -3.5969357, -6.4259725, -3.5982680, -2.1833124, 2.1754398
8: -2.0558996, 0.1803856, -2.0379558, 0.1801367, -1.7813401, 1.7580085
9: 2.4190264, 5.1630154, 2.4203515, 5.1597691, -2.2934179, 2.2881017

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6222

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3583368
time: 4.78 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3597421
time: 4.57 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -14.3501587, -10.2703190, -14.2979145, -10.2878590, -2.5760193, 2.5426273
1: -12.4914598, -8.9074078, -12.4860992, -8.9376917, -2.5631342, 2.5852952
2: -13.4113274, -10.1062279, -13.4084082, -10.1802959, -2.5448370, 2.5927935
3: -9.9437532, -6.8810663, -9.8884907, -6.9029832, -2.7612762, 2.7288766
4: -4.5657234, -2.3936605, -4.5532441, -2.4014409, -1.5914602, 1.5809147
5: -11.1370764, -7.3604593, -11.0722637, -7.3679972, -2.6155281, 2.5541754
6: -17.6468582, -13.6016884, -17.5752277, -13.6046658, -2.9232731, 2.8730030
7: -6.4431086, -3.5612941, -6.4322238, -3.5995317, -2.1990147, 2.2289383
8: -2.0776596, 0.1839032, -2.0386581, 0.1810880, -1.7980747, 1.7789836
9: 2.4166737, 5.1555319, 2.4199991, 5.1529918, -2.2901940, 2.2888691

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3590336, upper bound: 1.3490727
time: 5.18 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597367, upper bound: 1.3509654
time: 5.58 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -14.3549118, -10.2660093, -14.3000526, -10.2871857, -2.5811758, 2.5480752
1: -12.5077019, -8.9004421, -12.4945669, -8.9361744, -2.5745702, 2.6003580
2: -13.4156017, -10.1035509, -13.4097443, -10.1796179, -2.5512676, 2.5958443
3: -9.9479713, -6.8794489, -9.8902254, -6.9025607, -2.7658668, 2.7425418
4: -4.5802383, -2.3856285, -4.5608377, -2.3998196, -1.5946178, 1.5956216
5: -11.1407967, -7.3556747, -11.0733824, -7.3661222, -2.6229496, 2.5598645
6: -17.6569691, -13.5955963, -17.5802059, -13.6031609, -2.9308453, 2.8839712
7: -6.4474111, -3.5527673, -6.4331923, -3.5954514, -2.2093840, 2.2344120
8: -2.0828876, 0.1889706, -2.0398965, 0.1837621, -1.8062692, 1.7823820
9: 2.4053893, 5.1690454, 2.4171681, 5.1602287, -2.3086534, 2.2992454

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3590336, upper bound: 1.3578420
time: 5.39 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597367, upper bound: 1.3597361
time: 5.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.64 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3482352
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3496369
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3570050
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3584046
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3487952
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3496336
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3575662
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3574717, upper bound: 1.3584071
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3495653
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3583370, upper bound: 1.3509721
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3583368
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3583346, upper bound: 1.3597421
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3590336, upper bound: 1.3490727
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3597367, upper bound: 1.3509654
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3590336, upper bound: 1.3578420
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.64
Output dim: 9, lower bound: -1.3597367, upper bound: 1.3597361

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.2817593, -10.3078213, -2.4876318, 2.4765811
1: -12.4692907, -8.9608955, -12.4788408, -8.9498615, -2.5204434, 2.5197039
2: -13.3810949, -10.1954117, -13.3891401, -10.1910238, -2.4944372, 2.4989462
3: -9.8727856, -6.9312177, -9.8794422, -6.9286475, -2.6729813, 2.6722445
4: -4.5442271, -2.4259353, -4.5513921, -2.4133654, -1.5471497, 1.5432532
5: -11.0555973, -7.3901548, -11.0618057, -7.3864141, -2.5079269, 2.5116758
6: -17.5522270, -13.6359434, -17.5598240, -13.6208191, -2.8089590, 2.8029265
7: -6.4127755, -3.6144905, -6.4173303, -3.6078835, -2.1549206, 2.1551249
8: -2.0253696, 0.1595144, -2.0317454, 0.1670866, -1.7354312, 1.7342143
9: 2.4437881, 5.1446662, 2.4314313, 5.1514344, -2.2594266, 2.2638624

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560898, upper bound: 1.3474141
time: 4.74 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3482366
time: 5.15 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2752590, -10.3249445, -14.3030491, -10.2886257, -2.5156546, 2.4969969
1: -12.4692907, -8.9608955, -12.4860325, -8.9154034, -2.5624504, 2.5286040
2: -13.3810949, -10.1954117, -13.4031048, -10.1228971, -2.5450845, 2.5136404
3: -9.8727856, -6.9312177, -9.9350176, -6.8950319, -2.7076755, 2.7261226
4: -4.5442271, -2.4259353, -4.5596819, -2.4004102, -1.5598991, 1.5522146
5: -11.0555973, -7.3901548, -11.1264076, -7.3647404, -2.5294499, 2.5752397
6: -17.5522270, -13.6359434, -17.6045494, -13.6134529, -2.8421841, 2.8632984
7: -6.4127755, -3.6144905, -6.4396544, -3.5654535, -2.2020037, 2.1772428
8: -2.0253696, 0.1595144, -2.0580649, 0.1756454, -1.7445011, 1.7645726
9: 2.4437881, 5.1446662, 2.4179764, 5.1573458, -2.2668500, 2.2774174

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3487984
time: 4.82 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3496369
time: 4.85 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206253, -14.2838421, -10.3071423, -2.4926271, 2.4819655
1: -12.4855061, -8.9540024, -12.4873037, -8.9483461, -2.5318708, 2.5347271
2: -13.3854179, -10.1927128, -13.3904715, -10.1903477, -2.5008440, 2.5026164
3: -9.8770123, -6.9295988, -9.8811779, -6.9282260, -2.6774387, 2.6859150
4: -4.5587440, -2.4179046, -4.5589876, -2.4117429, -1.5505221, 1.5579810
5: -11.0593290, -7.3853664, -11.0629234, -7.3845363, -2.5173607, 2.5173759
6: -17.5622864, -13.6298618, -17.5647926, -13.6193142, -2.8170023, 2.8138881
7: -6.4170828, -3.6060100, -6.4183011, -3.6038098, -2.1652904, 2.1565576
8: -2.0306215, 0.1645761, -2.0329928, 0.1697617, -1.7437086, 1.7376189
9: 2.4325800, 5.1581740, 2.4286098, 5.1586723, -2.2779362, 2.2742362

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3561823
time: 7.57 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3570046
time: 4.82 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -14.2798958, -10.3206253, -14.3051624, -10.2879457, -2.5206504, 2.5024137
1: -12.4855061, -8.9540024, -12.4944983, -8.9138899, -2.5738735, 2.5436282
2: -13.3854179, -10.1927128, -13.4044390, -10.1222410, -2.5536685, 2.5173092
3: -9.8770123, -6.9295988, -9.9367485, -6.8946090, -2.7121344, 2.7341042
4: -4.5587440, -2.4179046, -4.5672750, -2.3987927, -1.5632699, 1.5669422
5: -11.0593290, -7.3853664, -11.1275234, -7.3628664, -2.5388756, 2.5803518
6: -17.5622864, -13.6298618, -17.6095200, -13.6119471, -2.8502259, 2.8742595
7: -6.4170828, -3.6060100, -6.4406209, -3.5613780, -2.2102952, 2.1786790
8: -2.0306215, 0.1645761, -2.0593147, 0.1783209, -1.7527809, 1.7677796
9: 2.4325800, 5.1581740, 2.4151545, 5.1645827, -2.2853601, 2.2877932

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 5816
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5816

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560921, upper bound: 1.3575666
time: 5.01 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3560897, upper bound: 1.3584044
time: 4.80 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2969637, -10.3056793, -14.2885551, -10.3204832, -2.4995050, 2.5132637
1: -12.4765329, -8.9240980, -12.4818211, -8.9513063, -2.5323772, 2.5676713
2: -13.3950863, -10.1241970, -13.3961096, -10.1862726, -2.5222206, 2.5647831
3: -9.9291859, -6.8954368, -9.8784199, -6.9062476, -2.7426133, 2.7042069
4: -4.5532007, -2.4129722, -4.5526085, -2.4164090, -1.5670228, 1.5597675
5: -11.1234884, -7.3681226, -11.0635071, -7.3699770, -2.5939307, 2.5333877
6: -17.5997391, -13.6285324, -17.5690498, -13.6301842, -2.8709106, 2.8486061
7: -6.4351735, -3.5703773, -6.4292521, -3.6048670, -2.1807103, 2.2122028
8: -2.0523562, 0.1680994, -2.0328236, 0.1685128, -1.7705884, 1.7576942
9: 2.4301715, 5.1506891, 2.4295969, 5.1517925, -2.2735088, 2.2734790

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557160, upper bound: 1.3487952
time: 5.00 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557160, upper bound: 1.3487957
time: 5.75 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2969637, -10.3056793, -14.3418980, -10.2851372, -2.5357018, 2.5393572
1: -12.4765329, -8.9240980, -12.4967737, -8.9348049, -2.5493283, 2.5833244
2: -13.3950863, -10.1241970, -13.4122791, -10.1675282, -2.5424924, 2.5764866
3: -9.9291859, -6.8954368, -9.8931093, -6.8915410, -2.7504201, 2.7191434
4: -4.5532007, -2.4129722, -4.5652409, -2.3972752, -1.5832705, 1.5720756
5: -11.1234884, -7.3681226, -11.0768089, -7.3622675, -2.5986075, 2.5473228
6: -17.5997391, -13.6285324, -17.6167450, -13.6033316, -2.8840609, 2.8813019
7: -6.4351735, -3.5703773, -6.4372034, -3.5958848, -2.1917162, 2.2221448
8: -2.0523562, 0.1680994, -2.0582781, 0.1843166, -1.7794337, 1.7761075
9: 2.4301715, 5.1506891, 2.4160953, 5.1566420, -2.2782323, 2.2867715

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 5747
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 5747
type: B, layer: 1, pos: 6222

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557160, upper bound: 1.3496337
time: 5.13 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3557160, upper bound: 1.3496335
time: 6.34 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3016415, -10.3013592, -14.2906160, -10.3198061, -2.5044937, 2.5186205
1: -12.4927559, -8.9171963, -12.4902830, -8.9498072, -2.5438051, 2.5826869
2: -13.3994026, -10.1215343, -13.3974438, -10.1855965, -2.5286293, 2.5678196
3: -9.9334030, -6.8938179, -9.8801527, -6.9058251, -2.7471981, 2.7178760
4: -4.5677137, -2.4049475, -4.5602021, -2.4147859, -1.5703976, 1.5744927
5: -11.1272068, -7.3633337, -11.0646267, -7.3681021, -2.6013470, 2.5390844
6: -17.6098232, -13.6224499, -17.5740223, -13.6286736, -2.8789549, 2.8595777
7: -6.4394732, -3.5618916, -6.4302206, -3.6008024, -2.1910834, 2.2176425
8: -2.0576110, 0.1731668, -2.0340729, 0.1711893, -1.7787843, 1.7611051
9: 2.4189587, 5.1641994, 2.4267778, 5.1590281, -2.2920113, 2.2838585

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3575657
time: 8.21 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487053, upper bound: 1.3575666
time: 5.21 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3016415, -10.3013592, -14.3440609, -10.2844667, -2.5406866, 2.5446293
1: -12.4927559, -8.9171963, -12.5052414, -8.9332790, -2.5607843, 2.5983467
2: -13.3994026, -10.1215343, -13.4136257, -10.1668453, -2.5489016, 2.5795319
3: -9.9334030, -6.8938179, -9.8948441, -6.8911200, -2.7550063, 2.7328124
4: -4.5677137, -2.4049475, -4.5728350, -2.3956451, -1.5864418, 1.5864146
5: -11.1272068, -7.3633337, -11.0779285, -7.3603969, -2.6060171, 2.5530238
6: -17.6098232, -13.6224499, -17.6217308, -13.6018200, -2.8916197, 2.8920290
7: -6.4394732, -3.5618916, -6.4381714, -3.5917964, -2.2021008, 2.2276006
8: -2.0576110, 0.1731668, -2.0595131, 0.1869888, -1.7876306, 1.7793186
9: 2.4189587, 5.1641994, 2.4132390, 5.1638808, -2.2967405, 2.2971933

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5747
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 6222
type: B, layer: 1, pos: 5747

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5747

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 833

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487053, upper bound: 1.3584036
time: 6.24 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487030, upper bound: 1.3584047
time: 5.80 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -14.3285809, -10.2895708, -14.2872143, -10.2886791, -2.5538969, 2.5026622
1: -12.4841938, -8.9443293, -12.4813242, -8.9417820, -2.5447865, 2.5340571
2: -13.3973303, -10.1770477, -13.3964005, -10.1877213, -2.5134406, 2.5274878
3: -9.8874311, -6.9168262, -9.8853741, -6.9267378, -2.6900129, 2.6928091
4: -4.5567980, -2.4067566, -4.5517287, -2.4045753, -1.5701332, 1.5620129
5: -11.0691080, -7.3824968, -11.0669537, -7.3852549, -2.5310488, 2.5266356
6: -17.5995598, -13.6090937, -17.5633507, -13.6058483, -2.8728623, 2.8235736
7: -6.4207535, -3.6054564, -6.4190893, -3.6047449, -2.1701789, 2.1694531
8: -2.0506783, 0.1753235, -2.0351710, 0.1744871, -1.7682657, 1.7527225
9: 2.4303093, 5.1495018, 2.4257817, 5.1521349, -2.2741122, 2.2748106

Time for backsubstitution: 14.72 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.3600305867793656, 1.360028066183916]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2406.78 seconds
