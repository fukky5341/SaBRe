## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.11948726150000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2823706, 0.2823703)
1: (-12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3489695, 0.3489695)
2: (-9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2716053, 0.2716053)
3: (-0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3234239, 0.3234239)
4: (-11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3780911, 0.3780909)
5: (7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537158, 0.2537158)
6: (-6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2523494, 0.2523494)
7: (-15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4796824, 0.4796824)
8: (-3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417318, 0.2417318)
9: (-3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122460, 0.3122458)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.56 + 35.06 = 57.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1200877, upper bound: 0.1200876

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 6232
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1183522
time: 6.76 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1200861
time: 4.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.99 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.99
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1183522
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.99
Output dim: 5, lower bound: -0.1200863, upper bound: 0.1200861

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.9960213, -10.2271366, -10.9987946, -10.2271118, -0.2762604, 0.2790488
1: -12.4132214, -11.6261654, -12.4154472, -11.6261301, -0.3440664, 0.3462827
2: -9.6395922, -8.8947325, -9.6398697, -8.8946791, -0.2708919, 0.2711487
3: -0.2386980, 0.5735416, -0.2439842, 0.5736139, -0.3100485, 0.3151636
4: -11.7314434, -10.8032789, -11.7315254, -10.8003359, -0.3727696, 0.3701000
5: 7.6976633, 8.3776760, 7.6976194, 8.3797541, -0.2498031, 0.2477565
6: -6.3843861, -5.5822525, -6.3855133, -5.5822067, -0.2489082, 0.2499772
7: -15.9063110, -14.9500103, -15.9111080, -14.9499550, -0.4689763, 0.4737647
8: -3.8060317, -3.1032228, -3.8060818, -3.1007190, -0.2386796, 0.2362202
9: -3.6104774, -2.9848332, -3.6108236, -2.9779658, -0.3033961, 0.2969317

Time for backsubstitution: 20.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183526, upper bound: 0.1183524
time: 6.34 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183526, upper bound: 0.1183524
time: 3.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.0022717, -10.2151604, -11.0019884, -10.2270823, -0.2789814, 0.2871244
1: -12.4187469, -11.6171675, -12.4180088, -11.6260939, -0.3471639, 0.3538572
2: -9.6405535, -8.8934698, -9.6402025, -8.8946161, -0.2715416, 0.2727973
3: -0.2502286, 0.5955524, -0.2500761, 0.5736964, -0.3178705, 0.3246539
4: -11.7436361, -10.7965784, -11.7316208, -10.7969465, -0.3819356, 0.3757172
5: 7.6885214, 8.3822784, 7.6975675, 8.3821468, -0.2569425, 0.2522024
6: -6.3868170, -5.5767164, -6.3868093, -5.5821552, -0.2515869, 0.2557016
7: -15.9174738, -14.9309082, -15.9166470, -14.9498930, -0.4740248, 0.4852827
8: -3.8162079, -3.0977001, -3.8061390, -3.0978241, -0.2454976, 0.2385113
9: -3.6389813, -2.9697738, -3.6112208, -2.9700487, -0.3157101, 0.3031733

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6232
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6232

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200741, upper bound: 0.1189507
time: 4.10 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200859, upper bound: 0.1200853
time: 4.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.79 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 29.79
Output dim: 5, lower bound: -0.1183526, upper bound: 0.1183524
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 29.79
Output dim: 5, lower bound: -0.1183526, upper bound: 0.1183524
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 5, lower bound: -0.1200741, upper bound: 0.1189507
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.79
Output dim: 5, lower bound: -0.1200859, upper bound: 0.1200853

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.0022449, -10.2181740, -11.0007515, -10.2333431, -0.2727079, 0.2798273
1: -12.4187164, -11.6205740, -12.4166079, -11.6331768, -0.3400476, 0.3437942
2: -9.6380491, -8.8937759, -9.6350117, -8.8960714, -0.2636154, 0.2675767
3: -0.2502286, 0.5922837, -0.2488117, 0.5670025, -0.3068473, 0.3174704
4: -11.7397757, -10.7966232, -11.7236328, -10.7985010, -0.3734438, 0.3667805
5: 7.6921558, 8.3822327, 7.7051072, 8.3806067, -0.2484539, 0.2446330
6: -6.3868175, -5.5792389, -6.3858271, -5.5873470, -0.2459126, 0.2502975
7: -15.9174004, -14.9326696, -15.9158230, -14.9535456, -0.4702222, 0.4811668
8: -3.8150148, -3.0977235, -3.8037500, -3.0982919, -0.2427654, 0.2324737
9: -3.6387806, -2.9739053, -3.6091797, -2.9786205, -0.3068998, 0.2966549

Time for backsubstitution: 21.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 6232
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4657

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200723, upper bound: 0.1182415
time: 3.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200723, upper bound: 0.1189487
time: 3.66 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.0022697, -10.2151585, -11.0019903, -10.2270803, -0.2727752, 0.2859228
1: -12.4187469, -11.6171665, -12.4180088, -11.6260967, -0.3407035, 0.3525120
2: -9.6405535, -8.8934708, -9.6402035, -8.8946161, -0.2715411, 0.2683225
3: -0.2502286, 0.5955565, -0.2500761, 0.5736983, -0.3121505, 0.3231611
4: -11.7436352, -10.7965765, -11.7316227, -10.7969456, -0.3803909, 0.3680031
5: 7.6885214, 8.3822765, 7.6975694, 8.3821478, -0.2554731, 0.2444897
6: -6.3868170, -5.5767155, -6.3868093, -5.5821557, -0.2463928, 0.2546847
7: -15.9174747, -14.9309101, -15.9166460, -14.9498940, -0.4705381, 0.4845831
8: -3.8162079, -3.0977006, -3.8061399, -3.0978236, -0.2448645, 0.2373044
9: -3.6389809, -2.9697747, -3.6112206, -2.9700480, -0.3070043, 0.3031371

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6232
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6232

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1200739
time: 4.33 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1200857
time: 3.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.27 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.27
Output dim: 5, lower bound: -0.1200723, upper bound: 0.1182415
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.27
Output dim: 5, lower bound: -0.1200723, upper bound: 0.1189487
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.27
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1200739
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.27
Output dim: 5, lower bound: -0.1189510, upper bound: 0.1200857

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.0012894, -10.2182293, -11.0005331, -10.2333565, -0.2713630, 0.2789707
1: -12.4187107, -11.6209679, -12.4166098, -11.6332674, -0.3393078, 0.3427089
2: -9.6363783, -8.8938026, -9.6346292, -8.8960772, -0.2619295, 0.2671647
3: -0.2498686, 0.5922294, -0.2487299, 0.5669894, -0.3064797, 0.3173504
4: -11.7388859, -10.7967768, -11.7234287, -10.7985363, -0.3727844, 0.3666196
5: 7.6921558, 8.3812122, 7.7051072, 8.3803768, -0.2478666, 0.2433699
6: -6.3868160, -5.5809450, -6.3858261, -5.5877342, -0.2453369, 0.2483772
7: -15.9153376, -14.9326725, -15.9153547, -14.9535446, -0.4676287, 0.4802740
8: -3.8150153, -3.0997524, -3.8037496, -3.0987511, -0.2421769, 0.2303267
9: -3.6384835, -2.9739053, -3.6091108, -2.9786205, -0.3062315, 0.2963930

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1182410
time: 4.21 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1182418
time: 3.77 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.0027294, -10.2141838, -11.0007448, -10.2333431, -0.2753692, 0.2799259
1: -12.4205437, -11.6198530, -12.4166088, -11.6331768, -0.3410647, 0.3446298
2: -9.6382055, -8.8851585, -9.6349983, -8.8960705, -0.2633128, 0.2702186
3: -0.2502260, 0.5943298, -0.2488122, 0.5670025, -0.3067654, 0.3176550
4: -11.7416391, -10.7917385, -11.7236280, -10.7985001, -0.3762983, 0.3700687
5: 7.6879363, 8.3826418, 7.7051072, 8.3806019, -0.2481728, 0.2460139
6: -6.3952146, -5.5789680, -6.3858271, -5.5873570, -0.2478166, 0.2502592
7: -15.9189978, -14.9223042, -15.9158077, -14.9535456, -0.4763432, 0.4809644
8: -3.8253078, -3.0972676, -3.8037500, -3.0982966, -0.2427819, 0.2332817
9: -3.6400428, -2.9718976, -3.6091774, -2.9786205, -0.3081768, 0.2965560

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189481
time: 4.90 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189490
time: 3.67 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.0010338, -10.2214222, -11.0019875, -10.2270832, -0.2757047, 0.2796478
1: -12.4173508, -11.6242447, -12.4180098, -11.6260967, -0.3418195, 0.3454016
2: -9.6353617, -8.8949242, -9.6402035, -8.8946285, -0.2663500, 0.2652462
3: -0.2489653, 0.5888584, -0.2500761, 0.5736930, -0.3108388, 0.3123060
4: -11.7356453, -10.7981339, -11.7316217, -10.7969465, -0.3715606, 0.3703135
5: 7.6960611, 8.3807373, 7.6975689, 8.3821430, -0.2479284, 0.2459333
6: -6.3858333, -5.5819082, -6.3868093, -5.5821571, -0.2482861, 0.2490303
7: -15.9166508, -14.9345608, -15.9166460, -14.9498930, -0.4730167, 0.4809217
8: -3.8138185, -3.0981674, -3.8061337, -3.0978231, -0.2390448, 0.2378068
9: -3.6369405, -2.9783459, -3.6112199, -2.9700480, -0.3061867, 0.2945641

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1200728
time: 5.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1200738
time: 3.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.0022707, -10.2151604, -11.0019903, -10.2270803, -0.2727749, 0.2808738
1: -12.4187460, -11.6171675, -12.4180088, -11.6260967, -0.3407037, 0.3472586
2: -9.6405544, -8.8934736, -9.6402035, -8.8946161, -0.2670670, 0.2683222
3: -0.2502286, 0.5955539, -0.2500761, 0.5736983, -0.3122543, 0.3188418
4: -11.7436352, -10.7965784, -11.7316227, -10.7969456, -0.3741996, 0.3680034
5: 7.6885214, 8.3822784, 7.6975694, 8.3821478, -0.2491999, 0.2444896
6: -6.3868170, -5.5767164, -6.3868093, -5.5821557, -0.2463927, 0.2504892
7: -15.9174757, -14.9309092, -15.9166460, -14.9498940, -0.4705379, 0.4817946
8: -3.8162069, -3.0977001, -3.8061399, -3.0978236, -0.2441149, 0.2373043
9: -3.6389809, -2.9697742, -3.6112206, -2.9700480, -0.3070441, 0.2944725

Time for backsubstitution: 21.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1189641
time: 4.91 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1189511
time: 4.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.60 seconds
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1182410
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1182418
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189481
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1183386, upper bound: 0.1189490
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1200728
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1200738
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1189641
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 5, lower bound: -0.1172172, upper bound: 0.1189511

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.0010338, -10.2214222, -10.9960203, -10.2271366, -0.2756065, 0.2735682
1: -12.4173508, -11.6242447, -12.4132185, -11.6261635, -0.3407979, 0.3405360
2: -9.6353617, -8.8949242, -9.6395931, -8.8947458, -0.2664614, 0.2645818
3: -0.2489653, 0.5888584, -0.2386980, 0.5735369, -0.3088644, 0.2997465
4: -11.7356453, -10.7981339, -11.7314396, -10.8032780, -0.3643374, 0.3685329
5: 7.6960611, 8.3807373, 7.6976643, 8.3776751, -0.2425057, 0.2445809
6: -6.3858333, -5.5819082, -6.3843865, -5.5822563, -0.2473171, 0.2459853
7: -15.9166508, -14.9345608, -15.9063072, -14.9500113, -0.4748690, 0.4703333
8: -3.8138185, -3.0981674, -3.8060293, -3.1032243, -0.2336023, 0.2392061
9: -3.6369405, -2.9783459, -3.6104770, -2.9848328, -0.2913487, 0.2948360

Time for backsubstitution: 20.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4657

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193637
time: 4.23 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200714
time: 3.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.0010338, -10.2214222, -11.0022688, -10.2151604, -0.2762972, 0.2734414
1: -12.4173508, -11.6242447, -12.4187460, -11.6171675, -0.3420169, 0.3433712
2: -9.6353617, -8.8949242, -9.6405544, -8.8934813, -0.2669137, 0.2653871
3: -0.2489653, 0.5888584, -0.2502286, 0.5955496, -0.3110017, 0.3070056
4: -11.7356453, -10.7981339, -11.7436323, -10.7965765, -0.3672023, 0.3706933
5: 7.6960611, 8.3807373, 7.6885228, 8.3822765, -0.2447374, 0.2460076
6: -6.3858333, -5.5819082, -6.3868175, -5.5767193, -0.2490255, 0.2466519
7: -15.9166508, -14.9345608, -15.9174747, -14.9309101, -0.4734199, 0.4707208
8: -3.8138185, -3.0981674, -3.8162041, -3.0977006, -0.2327361, 0.2380525
9: -3.6369405, -2.9783459, -3.6389809, -2.9697742, -0.2971749, 0.2950584

Time for backsubstitution: 20.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4657
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4657

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193645
time: 3.96 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200717
time: 4.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 29.40 seconds
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 29.40
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193637
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.40
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200714
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 29.40
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1193645
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.40
Output dim: 5, lower bound: -0.1172154, upper bound: 0.1200717

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -10.9960127, -10.2271376, -0.2760410, 0.2736654
1: -12.4191732, -11.6235256, -12.4132195, -11.6261673, -0.3402977, 0.3413545
2: -9.6355162, -8.8863087, -9.6395798, -8.8947468, -0.2661586, 0.2649448
3: -0.2489617, 0.5909069, -0.2386978, 0.5735366, -0.3087833, 0.2999364
4: -11.7375050, -10.7932510, -11.7314348, -10.8032770, -0.3671968, 0.3689640
5: 7.6918421, 8.3811483, 7.6976643, 8.3776684, -0.2422246, 0.2448632
6: -6.3942318, -5.5816393, -6.3843861, -5.5822630, -0.2471683, 0.2459476
7: -15.9182510, -14.9241943, -15.9062948, -14.9500113, -0.4777287, 0.4701302
8: -3.8241086, -3.0977125, -3.8060269, -3.1032281, -0.2336206, 0.2393231
9: -3.6382055, -2.9763370, -3.6104751, -2.9848328, -0.2926263, 0.2947371

Time for backsubstitution: 20.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4657

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165079, upper bound: 0.1200712
time: 4.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200712
time: 3.89 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -11.0022621, -10.2151604, -0.2767317, 0.2770342
1: -12.4191732, -11.6235256, -12.4187450, -11.6171694, -0.3415165, 0.3463387
2: -9.6355162, -8.8863087, -9.6405411, -8.8934813, -0.2666111, 0.2657502
3: -0.2489617, 0.5909069, -0.2502284, 0.5955503, -0.3109206, 0.3082810
4: -11.7375050, -10.7932510, -11.7436256, -10.7965784, -0.3700304, 0.3711250
5: 7.6918421, 8.3811483, 7.6885228, 8.3822689, -0.2472533, 0.2462897
6: -6.3942318, -5.5816393, -6.3868175, -5.5767279, -0.2488768, 0.2475120
7: -15.9182510, -14.9241943, -15.9174595, -14.9309092, -0.4784734, 0.4808013
8: -3.8241086, -3.0977125, -3.8162060, -3.0977077, -0.2365830, 0.2388477
9: -3.6382055, -2.9763370, -3.6389794, -2.9697742, -0.2987630, 0.2962999

Time for backsubstitution: 20.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4657
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4657

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200725
time: 4.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200720
time: 4.17 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 29.87 seconds
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 29.87
Output dim: 5, lower bound: -0.1165079, upper bound: 0.1200712
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.87
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200712
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 29.87
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200725
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.87
Output dim: 5, lower bound: -0.1165080, upper bound: 0.1200720

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -10.9950638, -10.2271910, -0.2751667, 0.2724366
1: -12.4191732, -11.6235256, -12.4132128, -11.6265574, -0.3393505, 0.3403828
2: -9.6355162, -8.8863087, -9.6379213, -8.8947735, -0.2664821, 0.2632623
3: -0.2489617, 0.5909069, -0.2383361, 0.5734820, -0.3087443, 0.2995751
4: -11.7375050, -10.7932510, -11.7305536, -10.8034315, -0.3671794, 0.3683081
5: 7.6918421, 8.3811483, 7.6976643, 8.3766556, -0.2410443, 0.2444068
6: -6.3942318, -5.5816393, -6.3843865, -5.5839610, -0.2452877, 0.2456599
7: -15.9182510, -14.9241943, -15.9042397, -14.9500113, -0.4769906, 0.4675813
8: -3.8241086, -3.0977125, -3.8060265, -3.1052504, -0.2314806, 0.2391881
9: -3.6382055, -2.9763370, -3.6101818, -2.9848328, -0.2922927, 0.2940948

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165077, upper bound: 0.1200413
time: 4.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1189477
time: 6.22 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -10.9965029, -10.2231398, -0.2766336, 0.2749119
1: -12.4191732, -11.6235256, -12.4150486, -11.6254444, -0.3418579, 0.3413675
2: -9.6355162, -8.8863087, -9.6397457, -8.8861275, -0.2666667, 0.2649784
3: -0.2489617, 0.5909069, -0.2386956, 0.5755830, -0.3089681, 0.2999364
4: -11.7375050, -10.7932510, -11.7333088, -10.7983942, -0.3676283, 0.3719493
5: 7.6918421, 8.3811483, 7.6934447, 8.3780842, -0.2430828, 0.2448633
6: -6.3942318, -5.5816393, -6.3927851, -5.5819831, -0.2477571, 0.2459474
7: -15.9182510, -14.9241943, -15.9078836, -14.9396467, -0.4777606, 0.4738207
8: -3.8241086, -3.0977125, -3.8163199, -3.1027756, -0.2343267, 0.2393678
9: -3.6382055, -2.9763370, -3.6117463, -2.9828248, -0.2926263, 0.2962408

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165076, upper bound: 0.1200412
time: 6.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200709
time: 4.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -11.0013132, -10.2152119, -0.2758577, 0.2757281
1: -12.4191732, -11.6235256, -12.4187393, -11.6175613, -0.3405695, 0.3432915
2: -9.6355162, -8.8863087, -9.6388798, -8.8935118, -0.2669344, 0.2640675
3: -0.2489617, 0.5909069, -0.2498686, 0.5954981, -0.3108815, 0.3078709
4: -11.7375050, -10.7932510, -11.7427435, -10.7967319, -0.3701582, 0.3704703
5: 7.6918421, 8.3811483, 7.6885228, 8.3812551, -0.2459996, 0.2458332
6: -6.3942318, -5.5816393, -6.3868160, -5.5784245, -0.2469963, 0.2467649
7: -15.9182510, -14.9241943, -15.9154081, -14.9309120, -0.4759104, 0.4782271
8: -3.8241086, -3.0977125, -3.8162026, -3.0997286, -0.2344431, 0.2385424
9: -3.6382055, -2.9763370, -3.6386828, -2.9697742, -0.2983465, 0.2956580

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165077, upper bound: 0.1200416
time: 4.89 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200713
time: 4.14 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.0015173, -10.2174273, -11.0027523, -10.2111654, -0.2773256, 0.2767026
1: -12.4191732, -11.6235256, -12.4205713, -11.6164474, -0.3430800, 0.3463515
2: -9.6355162, -8.8863087, -9.6407080, -8.8848667, -0.2671185, 0.2658283
3: -0.2489617, 0.5909069, -0.2502260, 0.5975990, -0.3111056, 0.3073448
4: -11.7375050, -10.7932510, -11.7454967, -10.7916946, -0.3725150, 0.3741211
5: 7.6918421, 8.3811483, 7.6843033, 8.3826857, -0.2461185, 0.2462897
6: -6.3942318, -5.5816393, -6.3952146, -5.5764465, -0.2494669, 0.2475110
7: -15.9182510, -14.9241943, -15.9190674, -14.9205465, -0.4785060, 0.4768710
8: -3.8241086, -3.0977125, -3.8264961, -3.0972471, -0.2335883, 0.2388912
9: -3.6382055, -2.9763370, -3.6402411, -2.9677668, -0.3000069, 0.2978085

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 53

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165076, upper bound: 0.1200424
time: 3.84 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200721
time: 4.12 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 29.11 seconds
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165077, upper bound: 0.1200413
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1189477
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165076, upper bound: 0.1200412
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200709
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165077, upper bound: 0.1200416
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200713
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165076, upper bound: 0.1200424
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 29.11
Output dim: 5, lower bound: -0.1165075, upper bound: 0.1200721

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.0013647, -10.2188473, -10.9950638, -10.2271910, -0.2749172, 0.2710605
1: -12.4187355, -11.6252613, -12.4132128, -11.6265574, -0.3392193, 0.3384833
2: -9.6343212, -8.8867264, -9.6379213, -8.8947735, -0.2645919, 0.2622671
3: -0.2486210, 0.5875108, -0.2383361, 0.5734820, -0.3085788, 0.2964761
4: -11.7367840, -10.7943668, -11.7305536, -10.8034315, -0.3664305, 0.3672452
5: 7.6925817, 8.3810244, 7.6976643, 8.3766556, -0.2405027, 0.2442203
6: -6.3937397, -5.5828638, -6.3843865, -5.5839610, -0.2446041, 0.2445390
7: -15.9177008, -14.9275599, -15.9042397, -14.9500113, -0.4760648, 0.4643242
8: -3.8238511, -3.0985184, -3.8060265, -3.1052504, -0.2308753, 0.2380006
9: -3.6370287, -2.9764893, -3.6101818, -2.9848328, -0.2901254, 0.2931230

Time for backsubstitution: 20.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 53

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1164780, upper bound: 0.1200409
time: 4.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1164780, upper bound: 0.1200409
time: 5.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.0013647, -10.2188473, -10.9965029, -10.2231398, -0.2763842, 0.2735360
1: -12.4187355, -11.6252613, -12.4150486, -11.6254444, -0.3417273, 0.3394675
2: -9.6343212, -8.8867264, -9.6397457, -8.8861275, -0.2647769, 0.2639824
3: -0.2486210, 0.5875108, -0.2386956, 0.5755830, -0.3088026, 0.2968373
4: -11.7367840, -10.7943668, -11.7333088, -10.7983942, -0.3668797, 0.3708864
5: 7.6925817, 8.3810244, 7.6934447, 8.3780842, -0.2425412, 0.2446768
6: -6.3937397, -5.5828638, -6.3927851, -5.5819831, -0.2470735, 0.2448262
7: -15.9177008, -14.9275599, -15.9078836, -14.9396467, -0.4768336, 0.4705634
8: -3.8238511, -3.0985184, -3.8163199, -3.1027756, -0.2337213, 0.2381802
9: -3.6370287, -2.9764893, -3.6117463, -2.9828248, -0.2904584, 0.2952689

Time for backsubstitution: 20.41 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.63 + 543.68 = 601.31 seconds
