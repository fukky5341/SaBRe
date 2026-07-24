## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.32578885785


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4069926, 1.4069926)
1: (-3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9955665, 1.9955665)
2: (-1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378)
3: (-1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8541119, 0.8541120)
4: (-0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7479159, 0.7479158)
5: (-1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7824583, 0.7824583)
6: (-0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135126, 0.4135125)
7: (-2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8645149, 0.8645149)
8: (-3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1990520, 1.1990521)
9: (-3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0880301, 1.0880303)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.06 + 98.42 = 103.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.3268303, upper bound: 0.3268329

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 834

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3246

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265782, upper bound: 0.3265765
time: 874.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3267455, upper bound: 0.3265788
time: 440.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1314.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1314.62
Output dim: 0, lower bound: -0.3265782, upper bound: 0.3265765
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1314.62
Output dim: 0, lower bound: -0.3267455, upper bound: 0.3265788

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4035972, 1.4036987
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9923661, 1.9924710
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8528330, 0.8527396
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7472535, 0.7472456
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7800068, 0.7798392
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135964, 0.4136108
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8616796, 0.8615350
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1966133, 1.1967012
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0866405, 1.0866737

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2721

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2986

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265190, upper bound: 0.3266856
time: 51.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265101, upper bound: 0.3266959
time: 17.82 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4036987, 1.4035972
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9924710, 1.9923661
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8527396, 0.8528330
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7472456, 0.7472535
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7798392, 0.7800068
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4136109, 0.4135964
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8615348, 0.8616796
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1967013, 1.1966134
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0866737, 1.0866406

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2986

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265260, upper bound: 0.3263808
time: 245.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265518, upper bound: 0.3263628
time: 327.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 576.61 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 576.61
Output dim: 0, lower bound: -0.3265190, upper bound: 0.3266856
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 576.61
Output dim: 0, lower bound: -0.3265101, upper bound: 0.3266959
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 576.61
Output dim: 0, lower bound: -0.3265260, upper bound: 0.3263808
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 576.61
Output dim: 0, lower bound: -0.3265518, upper bound: 0.3263628

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4034032, 1.4034567
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9919344, 1.9920171
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8528290, 0.8527356
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7470232, 0.7470103
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7800020, 0.7798345
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4136424, 0.4136559
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8616865, 0.8615397
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1944199, 1.1943852
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0856619, 1.0856779

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 2902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264891, upper bound: 0.3266502
time: 74.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264767, upper bound: 0.3266580
time: 386.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4033554, 1.4035045
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9919122, 1.9920393
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8528292, 0.8527355
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7470183, 0.7470152
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7800021, 0.7798344
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4136414, 0.4136569
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8616843, 0.8615419
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1942973, 1.1945077
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0856446, 1.0856953

Time for backsubstitution: 4.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2909

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 504

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265121, upper bound: 0.3265859
time: 25.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264062, upper bound: 0.3266908
time: 532.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4030664, 1.4029967
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9908639, 1.9904532
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8526146, 0.8526813
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7467990, 0.7469701
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7798212, 0.7799884
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135736, 0.4135738
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8610870, 0.8612995
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1951040, 1.1944883
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0850008, 1.0844249

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2708

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265205, upper bound: 0.3263835
time: 192.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265280, upper bound: 0.3263808
time: 56.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4030981, 1.4029649
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9905580, 1.9907591
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8525879, 0.8527080
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7469622, 0.7468070
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7798209, 0.7799888
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135883, 0.4135591
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8611549, 0.8612316
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1945763, 1.1950161
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0844581, 1.0849674

Time for backsubstitution: 4.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265249, upper bound: 0.3262143
time: 168.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264064, upper bound: 0.3263377
time: 60.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 233.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3264891, upper bound: 0.3266502
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3264767, upper bound: 0.3266580
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3265121, upper bound: 0.3265859
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3264062, upper bound: 0.3266908
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3265205, upper bound: 0.3263835
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3265280, upper bound: 0.3263808
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3265249, upper bound: 0.3262143
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 233.59
Output dim: 0, lower bound: -0.3264064, upper bound: 0.3263377

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4026587, 1.4027033
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9910834, 1.9911329
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8525690, 0.8524568
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7469942, 0.7469826
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7799610, 0.7797803
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135191, 0.4135414
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8612834, 0.8611363
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1924244, 1.1923443
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0848225, 1.0848014

Time for backsubstitution: 4.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 770

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263998, upper bound: 0.3261954
time: 148.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3260782, upper bound: 0.3265849
time: 486.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4026496, 1.4027123
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9910502, 1.9911660
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8525501, 0.8524756
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7469955, 0.7469813
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7799478, 0.7797935
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135279, 0.4135326
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8612831, 0.8611366
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1923789, 1.1923896
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0847853, 1.0848386

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 3054

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2851

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264372, upper bound: 0.3266083
time: 618.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264311, upper bound: 0.3266256
time: 126.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4033551, 1.4035056
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9919124, 1.9920392
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8528401, 0.8527158
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7469993, 0.7470046
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7799976, 0.7798334
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4136170, 0.4136510
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8615862, 0.8613184
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1942689, 1.1944371
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0856347, 1.0856793

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2950

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3004

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263883, upper bound: 0.3260576
time: 273.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3261601, upper bound: 0.3264545
time: 31.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4033566, 1.4035043
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9919122, 1.9920394
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8528095, 0.8527466
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7470076, 0.7469962
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7800010, 0.7798299
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4136355, 0.4136325
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8614609, 0.8614437
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1942269, 1.1944793
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0856290, 1.0856853

Time for backsubstitution: 4.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 3527

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2750

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263885, upper bound: 0.3266534
time: 294.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263641, upper bound: 0.3266860
time: 13.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4030790, 1.4030173
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9904072, 1.9899826
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8525426, 0.8526017
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7464366, 0.7466283
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7798115, 0.7799788
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4132993, 0.4133088
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8606986, 0.8608987
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1946269, 1.1939968
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0840390, 1.0834349

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2408

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 586

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264897, upper bound: 0.3263773
time: 88.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265141, upper bound: 0.3263517
time: 252.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4030871, 1.4030092
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9903934, 1.9899967
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8525349, 0.8526094
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7464571, 0.7466078
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7798115, 0.7799788
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4133086, 0.4132996
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8606861, 0.8609113
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1946123, 1.1940113
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0840108, 1.0834631

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264858, upper bound: 0.3262404
time: 94.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263906, upper bound: 0.3263411
time: 21.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4030193, 1.4028676
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9904898, 1.9906702
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8525438, 0.8526688
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7469572, 0.7468014
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797723, 0.7799474
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135755, 0.4135490
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8611503, 0.8612264
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1944584, 1.1948743
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0844150, 1.0849204

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2712

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 846

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3265192, upper bound: 0.3260644
time: 477.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263807, upper bound: 0.3262112
time: 61.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4030007, 1.4028862
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9904691, 1.9906909
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8525486, 0.8526641
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7469566, 0.7468020
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7797795, 0.7799401
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135783, 0.4135463
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8611498, 0.8612270
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1944343, 1.1948981
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0844111, 1.0849243

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 2986
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2850
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 3177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 886

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264049, upper bound: 0.3263338
time: 82.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3264049, upper bound: 0.3263396
time: 33.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 120.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3263998, upper bound: 0.3261954
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3260782, upper bound: 0.3265849
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3264372, upper bound: 0.3266083
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3264311, upper bound: 0.3266256
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3263883, upper bound: 0.3260576
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3261601, upper bound: 0.3264545
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3263885, upper bound: 0.3266534
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3263641, upper bound: 0.3266860
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3264897, upper bound: 0.3263773
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3265141, upper bound: 0.3263517
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3264858, upper bound: 0.3262404
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3263906, upper bound: 0.3263411
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3265192, upper bound: 0.3260644
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3263807, upper bound: 0.3262112
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3264049, upper bound: 0.3263338
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 120.66
Output dim: 0, lower bound: -0.3264049, upper bound: 0.3263396

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -3.0209565, -0.9632084, -3.0209565, -0.9632084, -1.4016676, 1.4016788
1: -3.8853817, -0.9901266, -3.8853817, -0.9901266, -1.9898417, 1.9898531
2: -1.0377467, -0.2949088, -1.0377467, -0.2949088, -0.7428378, 0.7428378
3: -1.3973571, -0.2312617, -1.3973571, -0.2312617, -0.8519503, 0.8518411
4: -0.7051620, 0.3348906, -0.7051620, 0.3348906, -0.7467837, 0.7467797
5: -1.6625407, -0.6227128, -1.6625407, -0.6227128, -0.7793815, 0.7792028
6: -0.1922126, 0.5676912, -0.1922126, 0.5676912, -0.4135039, 0.4135269
7: -2.1631868, -1.0505955, -2.1631868, -1.0505955, -0.8612825, 0.8611355
8: -3.1872282, -0.9482212, -3.1872282, -0.9482212, -1.1921389, 1.1920376
9: -3.9053702, -1.7991552, -3.9053702, -1.7991552, -1.0842776, 1.0842373

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3191
type: RSZ, layer: 1, pos: 2752
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 2718
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2629
type: RSZ, layer: 1, pos: 2719
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2919
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 3513
type: RSZ, layer: 1, pos: 3527
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2895
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3004
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 2865
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2767
type: RSZ, layer: 1, pos: 2657
type: RSZ, layer: 1, pos: 2711
type: RSZ, layer: 1, pos: 2823
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 2745
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3342
type: RSZ, layer: 1, pos: 3526
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 374
type: RSZ, layer: 1, pos: 3059
type: RSZ, layer: 1, pos: 3177
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2717
type: RSZ, layer: 1, pos: 504
type: RSZ, layer: 1, pos: 2503
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 2973
type: RSZ, layer: 1, pos: 341
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2836
type: RSZ, layer: 1, pos: 2820
type: RSZ, layer: 1, pos: 2917
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2704
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2369
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2773
type: RSZ, layer: 1, pos: 2644
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2311
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2700
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3525
type: RSZ, layer: 1, pos: 2914
type: RSZ, layer: 1, pos: 2708
type: RSZ, layer: 1, pos: 3295
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 2703
type: RSZ, layer: 1, pos: 2133
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2732
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2715
type: RSZ, layer: 1, pos: 2761
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2749
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2488
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 2416
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 3294
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2880
type: RSZ, layer: 1, pos: 2918
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 3350
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 2819
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 3314
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 2707
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 3164
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2881
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2132
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2415
type: RSZ, layer: 1, pos: 2825
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2308
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 2754
type: RSZ, layer: 1, pos: 2765
type: RSZ, layer: 1, pos: 3538
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 2733
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 416
type: RSZ, layer: 1, pos: 2818
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2753
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2864
type: RSZ, layer: 1, pos: 431
type: RSZ, layer: 1, pos: 3574
type: RSZ, layer: 1, pos: 3497
type: RSZ, layer: 1, pos: 343
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2706
type: RSZ, layer: 1, pos: 2822
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2789
type: RSZ, layer: 1, pos: 2908
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2672
type: RSZ, layer: 1, pos: 2746
type: RSZ, layer: 1, pos: 2534
type: RSZ, layer: 1, pos: 3443
type: RSZ, layer: 1, pos: 2710
type: RSZ, layer: 1, pos: 2698
type: RSZ, layer: 1, pos: 2844
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2712
type: RSZ, layer: 1, pos: 3570
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2734
type: RSZ, layer: 1, pos: 2949
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2721
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 2803
type: RSZ, layer: 1, pos: 2720
type: RSZ, layer: 1, pos: 2851
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2751
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 3595
type: RSZ, layer: 1, pos: 2701
type: RSZ, layer: 1, pos: 2983
type: RSZ, layer: 1, pos: 2716
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 3493
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 2730
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2119
type: RSZ, layer: 1, pos: 3040
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2120
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3354
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2760
type: RSZ, layer: 1, pos: 2852
type: RSZ, layer: 1, pos: 2835
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2747
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 2821
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2837
type: RSZ, layer: 1, pos: 2731
type: RSZ, layer: 1, pos: 2705
type: RSZ, layer: 1, pos: 3597
type: RSZ, layer: 1, pos: 2879
type: RSZ, layer: 1, pos: 2702
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 3569
type: RSZ, layer: 1, pos: 2907
type: RSZ, layer: 1, pos: 3247
type: RSZ, layer: 1, pos: 2709
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2909
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 2766
type: RSZ, layer: 1, pos: 2902
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 2736
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2838
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 2782
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 2804
type: RSZ, layer: 1, pos: 2972
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2992
type: RSZ, layer: 1, pos: 3090
type: RSZ, layer: 1, pos: 3403
type: RSZ, layer: 1, pos: 3404
type: RSZ, layer: 1, pos: 3178
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2748
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2924
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 389
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2750
type: RSZ, layer: 1, pos: 3568
type: RSZ, layer: 1, pos: 2725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3191

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263564, upper bound: 0.3261683
time: 190.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3263777, upper bound: 0.3261502
time: 372.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 566.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 566.63
Output dim: 0, lower bound: -0.3263564, upper bound: 0.3261683
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 566.63
Output dim: 0, lower bound: -0.3263777, upper bound: 0.3261502
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3260782, upper bound: 0.3265849
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3264372, upper bound: 0.3266083
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3264311, upper bound: 0.3266256
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3263883, upper bound: 0.3260576
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3261601, upper bound: 0.3264545
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3263885, upper bound: 0.3266534
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3263641, upper bound: 0.3266860
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3264897, upper bound: 0.3263773
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3265141, upper bound: 0.3263517
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3264858, upper bound: 0.3262404
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3263906, upper bound: 0.3263411
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3265192, upper bound: 0.3260644
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3263807, upper bound: 0.3262112
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3264049, upper bound: 0.3263338
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 566.63
Output dim: 0, lower bound: -0.3264049, upper bound: 0.3263396

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 103.48 + 7185.79 = 7289.27 seconds
