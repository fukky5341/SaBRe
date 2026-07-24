## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.016598636200000003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100981, 0.4100981)
1: (-4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186764, 0.5186763)
2: (-0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348959, 0.0348959)
3: (-0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294576, 0.1294576)
4: (-0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249279, 0.1249279)
5: (-0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541328, 0.1541328)
6: (0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0408004, 0.0408004)
7: (-0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171396, 0.3171396)
8: (-5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570098, 0.3570098)
9: (-4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603758, 0.3603759)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.93 + 17.87 = 25.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0166300, upper bound: 0.0166398

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3062

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166061, upper bound: 0.0166200
time: 5.37 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166174, upper bound: 0.0166098
time: 18.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 24.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 24.24
Output dim: 4, lower bound: -0.0166061, upper bound: 0.0166200
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 24.24
Output dim: 4, lower bound: -0.0166174, upper bound: 0.0166098

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100972, 0.4100972
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186732, 0.5186727
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348942, 0.0348943
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294575, 0.1294575
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249276, 0.1249276
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541320, 0.1541320
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407998, 0.0407998
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171377, 0.3171377
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570091, 0.3570091
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603748, 0.3603747

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2388

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165716, upper bound: 0.0166158
time: 46.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166035, upper bound: 0.0165876
time: 11.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100972, 0.4100972
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186727, 0.5186732
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348943, 0.0348942
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294575, 0.1294575
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249276, 0.1249276
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541320, 0.1541320
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407998, 0.0407998
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171377, 0.3171377
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570091, 0.3570091
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603746, 0.3603748

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2388

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165830, upper bound: 0.0166048
time: 11.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166158, upper bound: 0.0165721
time: 101.88 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 119.77 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 119.77
Output dim: 4, lower bound: -0.0165716, upper bound: 0.0166158
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 119.77
Output dim: 4, lower bound: -0.0166035, upper bound: 0.0165876
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 119.77
Output dim: 4, lower bound: -0.0165830, upper bound: 0.0166048
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 119.77
Output dim: 4, lower bound: -0.0166158, upper bound: 0.0165721

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100925, 0.4100926
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186481, 0.5186474
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348969, 0.0348970
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294598, 0.1294597
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249250, 0.1249250
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541351, 0.1541350
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0408013, 0.0408013
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171302, 0.3171301
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570062, 0.3570062
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603705, 0.3603703

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3063

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165630, upper bound: 0.0166211
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165671, upper bound: 0.0166019
time: 54.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100925, 0.4100924
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186480, 0.5186474
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348969, 0.0348969
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294597, 0.1294598
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249250, 0.1249250
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541350, 0.1541352
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0408013, 0.0408013
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171302, 0.3171301
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570062, 0.3570062
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603705, 0.3603703

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3063

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165940, upper bound: 0.0165824
time: 103.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165990, upper bound: 0.0165697
time: 28.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100925, 0.4100924
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186476, 0.5186479
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348970, 0.0348969
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294598, 0.1294597
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249250, 0.1249250
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541351, 0.1541349
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0408013, 0.0408013
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171302, 0.3171301
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570062, 0.3570062
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603703, 0.3603705

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3063

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165684, upper bound: 0.0166060
time: 54.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165814, upper bound: 0.0166011
time: 2.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100925, 0.4100924
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186474, 0.5186481
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348970, 0.0348969
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294597, 0.1294599
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249250, 0.1249250
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541350, 0.1541351
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0408013, 0.0408013
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171302, 0.3171301
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570062, 0.3570062
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603703, 0.3603705

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3063

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165994, upper bound: 0.0165749
time: 25.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166133, upper bound: 0.0165695
time: 2.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.50 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0165630, upper bound: 0.0166211
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0165671, upper bound: 0.0166019
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0165940, upper bound: 0.0165824
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0165990, upper bound: 0.0165697
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0165684, upper bound: 0.0166060
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0165814, upper bound: 0.0166011
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0165994, upper bound: 0.0165749
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.50
Output dim: 4, lower bound: -0.0166133, upper bound: 0.0165695

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100966, 0.4100966
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186262, 0.5186245
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348913, 0.0348914
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294610, 0.1294608
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249240, 0.1249241
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541345, 0.1541343
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407995, 0.0407995
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171217, 0.3171221
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570083, 0.3570085
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603643, 0.3603640

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165510, upper bound: 0.0166094
time: 41.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165572, upper bound: 0.0166053
time: 11.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100966, 0.4100966
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186253, 0.5186255
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348913, 0.0348913
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294610, 0.1294608
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249240, 0.1249241
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541345, 0.1541343
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407995, 0.0407995
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171220, 0.3171219
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570083, 0.3570085
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603642, 0.3603641

Time for backsubstitution: 6.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165576, upper bound: 0.0165966
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165617, upper bound: 0.0165929
time: 53.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100966, 0.4100966
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186251, 0.5186255
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348913, 0.0348913
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294608, 0.1294611
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249241, 0.1249240
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541343, 0.1541345
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407995, 0.0407995
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171220, 0.3171217
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570083, 0.3570085
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603641, 0.3603641

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165893, upper bound: 0.0165684
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165937, upper bound: 0.0165597
time: 59.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100966, 0.4100966
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186256, 0.5186250
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348913, 0.0348913
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294610, 0.1294608
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249241, 0.1249241
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541345, 0.1541343
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407995, 0.0407995
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171220, 0.3171220
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570083, 0.3570085
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603642, 0.3603641

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165567, upper bound: 0.0165942
time: 92.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165618, upper bound: 0.0165959
time: 24.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100966, 0.4100966
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186247, 0.5186260
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348914, 0.0348913
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294609, 0.1294608
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249241, 0.1249241
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541345, 0.1541343
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407995, 0.0407995
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171222, 0.3171217
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570083, 0.3570083
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603640, 0.3603643

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165720, upper bound: 0.0165994
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165738, upper bound: 0.0165907
time: 2.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100966, 0.4100966
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186254, 0.5186253
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348913, 0.0348913
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294608, 0.1294609
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249241, 0.1249240
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541343, 0.1541345
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407995, 0.0407995
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171220, 0.3171219
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570083, 0.3570083
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603641, 0.3603641

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165879, upper bound: 0.0165670
time: 38.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165936, upper bound: 0.0165646
time: 8.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100968, 0.4100966
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5186245, 0.5186262
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348914, 0.0348912
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294608, 0.1294611
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1249241, 0.1249240
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541343, 0.1541345
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407995, 0.0407995
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171222, 0.3171217
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570083, 0.3570083
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603640, 0.3603643

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3078

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166036, upper bound: 0.0165676
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166058, upper bound: 0.0165535
time: 49.01 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 57.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165510, upper bound: 0.0166094
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165572, upper bound: 0.0166053
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165576, upper bound: 0.0165966
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165617, upper bound: 0.0165929
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165893, upper bound: 0.0165684
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165937, upper bound: 0.0165597
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165567, upper bound: 0.0165942
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165618, upper bound: 0.0165959
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165720, upper bound: 0.0165994
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165738, upper bound: 0.0165907
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165879, upper bound: 0.0165670
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0165936, upper bound: 0.0165646
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0166036, upper bound: 0.0165676
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 57.45
Output dim: 4, lower bound: -0.0166058, upper bound: 0.0165535

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100732, 0.4100726
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5185403, 0.5185359
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348875, 0.0348877
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294737, 0.1294730
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248598, 0.1248623
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541504, 0.1541495
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407993, 0.0407994
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171086, 0.3171093
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570057, 0.3570056
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603358, 0.3603346

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3092

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0164981, upper bound: 0.0166119
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165493, upper bound: 0.0165622
time: 23.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100725, 0.4100733
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5185374, 0.5185392
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348876, 0.0348876
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294732, 0.1294735
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248622, 0.1248598
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541497, 0.1541502
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407993, 0.0407993
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171091, 0.3171090
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570057, 0.3570056
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603350, 0.3603356

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3092

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0165094, upper bound: 0.0166106
time: 132.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165555, upper bound: 0.0165565
time: 45.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100732, 0.4100726
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5185391, 0.5185373
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348876, 0.0348876
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294736, 0.1294731
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248598, 0.1248623
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541504, 0.1541495
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407993, 0.0407993
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171089, 0.3171091
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570057, 0.3570056
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603356, 0.3603350

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3092

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165195, upper bound: 0.0165927
time: 135.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165702, upper bound: 0.0165128
time: 46.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100732, 0.4100726
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5185391, 0.5185373
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348876, 0.0348876
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294735, 0.1294732
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248598, 0.1248623
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541502, 0.1541497
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407993, 0.0407993
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171090, 0.3171091
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570057, 0.3570056
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603355, 0.3603350

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3092

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165513, upper bound: 0.0165633
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166021, upper bound: 0.0165044
time: 20.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4100726, 0.4100733
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5185357, 0.5185401
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348877, 0.0348875
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294730, 0.1294737
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1248622, 0.1248598
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1541495, 0.1541504
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407994, 0.0407993
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3171093, 0.3171086
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3570057, 0.3570056
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3603346, 0.3603358

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3092

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165581, upper bound: 0.0165576
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166042, upper bound: 0.0165021
time: 79.28 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 87.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0164981, upper bound: 0.0166119
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0165493, upper bound: 0.0165622
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0165094, upper bound: 0.0166106
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0165555, upper bound: 0.0165565
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0165195, upper bound: 0.0165927
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0165702, upper bound: 0.0165128
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0165513, upper bound: 0.0165633
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0166021, upper bound: 0.0165044
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0165581, upper bound: 0.0165576
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 87.55
Output dim: 4, lower bound: -0.0166042, upper bound: 0.0165021

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4097326, 0.4096936
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5173043, 0.5171769
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348793, 0.0348790
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294011, 0.1293939
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245312, 0.1245609
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1540969, 0.1540912
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407988, 0.0407989
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170859, 0.3170871
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3552400, 0.3550617
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3587510, 0.3585936

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0164567, upper bound: 0.0166106
time: 45.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164914, upper bound: 0.0165633
time: 11.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4097320, 0.4096942
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5173022, 0.5171802
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348794, 0.0348789
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294009, 0.1293944
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245337, 0.1245590
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1540965, 0.1540920
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407988, 0.0407988
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170862, 0.3170868
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3552399, 0.3550617
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3587505, 0.3585945

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0164798, upper bound: 0.0166100
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165025, upper bound: 0.0165531
time: 18.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4096942, 0.4097320
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5171801, 0.5173023
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348789, 0.0348794
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293944, 0.1294009
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245590, 0.1245337
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1540920, 0.1540965
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407988, 0.0407988
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170868, 0.3170862
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3550617, 0.3552399
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3585945, 0.3587505

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165437, upper bound: 0.0165110
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166000, upper bound: 0.0164811
time: 62.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4096936, 0.4097326
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5171767, 0.5173042
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348790, 0.0348793
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293939, 0.1294011
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245609, 0.1245312
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1540913, 0.1540969
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407989, 0.0407988
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170871, 0.3170859
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3550617, 0.3552400
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3585936, 0.3587509

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3093

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165573, upper bound: 0.0165006
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0166021, upper bound: 0.0164587
time: 27.83 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 37.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0164567, upper bound: 0.0166106
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0164914, upper bound: 0.0165633
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0164798, upper bound: 0.0166100
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0165025, upper bound: 0.0165531
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0165437, upper bound: 0.0165110
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0166000, upper bound: 0.0164811
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0165573, upper bound: 0.0165006
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 37.37
Output dim: 4, lower bound: -0.0166021, upper bound: 0.0164587

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4097289, 0.4096702
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5172887, 0.5170343
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348786, 0.0348777
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1294001, 0.1293883
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1244929, 0.1245526
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1540968, 0.1540890
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407987, 0.0407988
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170832, 0.3170859
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3552669, 0.3549250
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3587412, 0.3584268

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0163975, upper bound: 0.0165929
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164399, upper bound: 0.0165548
time: 3.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4097283, 0.4096708
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5172870, 0.5170376
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348787, 0.0348776
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293999, 0.1293888
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1244954, 0.1245511
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1540965, 0.1540897
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407988, 0.0407988
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170837, 0.3170856
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3552669, 0.3549250
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3587409, 0.3584278

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2417

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164239, upper bound: 0.0165875
time: 8.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0164629, upper bound: 0.0165412
time: 16.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4281197, -3.7347143, -4.4281197, -3.7347143, -0.4096708, 0.4097283
1: -4.9976182, -4.0675087, -4.9976182, -4.0675087, -0.5170379, 0.5172870
2: -0.5030074, -0.2839484, -0.5030074, -0.2839484, -0.0348776, 0.0348787
3: -0.5186703, -0.3154251, -0.5186703, -0.3154251, -0.1293888, 0.1293999
4: -0.2425989, 0.0868944, -0.2425989, 0.0868944, -0.1245511, 0.1244954
5: -0.9782381, -0.6966124, -0.9782381, -0.6966124, -0.1540897, 0.1540965
6: 0.3157382, 0.5166167, 0.3157382, 0.5166167, -0.0407988, 0.0407988
7: -0.9941720, -0.5435872, -0.9941720, -0.5435872, -0.3170856, 0.3170837
8: -5.7481003, -5.1337290, -5.7481003, -5.1337290, -0.3549250, 0.3552669
9: -4.4941902, -3.9321339, -4.4941902, -3.9321339, -0.3584278, 0.3587409

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2334
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2827
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3293

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2417

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165425, upper bound: 0.0164636
time: 65.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0165835, upper bound: 0.0164336
time: 2.51 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 74.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 74.41
Output dim: 4, lower bound: -0.0163975, upper bound: 0.0165929
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 74.41
Output dim: 4, lower bound: -0.0164399, upper bound: 0.0165548
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 74.41
Output dim: 4, lower bound: -0.0164239, upper bound: 0.0165875
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 74.41
Output dim: 4, lower bound: -0.0164629, upper bound: 0.0165412
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 74.41
Output dim: 4, lower bound: -0.0165425, upper bound: 0.0164636
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 74.41
Output dim: 4, lower bound: -0.0165835, upper bound: 0.0164336
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 74.41
Output dim: 4, lower bound: -0.0166021, upper bound: 0.0164587

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 25.80 + 1782.10 = 1807.90 seconds
