## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 7200 seconds
Split limit: 100
Threshold: 44.215418322


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493)
1: (-25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925)
2: (-21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020)
3: (-24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430)
4: (-28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984)
5: (-24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609)
6: (-54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116)
7: (-30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833)
8: (-36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536)
9: (-29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944)
10: (-49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793)
11: (-49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476)
12: (-55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7024841, 79.7024841)
13: (-50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531)
14: (-87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285)
15: (-35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889)
16: (-46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353)
17: (-85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725)
18: (-49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497)
19: (-39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108)
20: (-37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752)
21: (-48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026)
22: (-50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610)
23: (-39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382)
24: (-46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789)
25: (-41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755)
26: (-57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847)
27: (-45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949)
28: (-39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258)
29: (-51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005)
30: (-49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708)
31: (-51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743)
32: (-52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832)
33: (-72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0874100, 106.0874023)
34: (-65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9840164, 81.9840240)
35: (-63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9368591, 85.9368591)
36: (-62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077)
37: (-87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498)
38: (-70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649)
39: (-80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968)
40: (-62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332)
41: (-55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251)
42: (-36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.89 + 101.49 = 104.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -44.2596780, upper bound: 44.2596780

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 650

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2178112, upper bound: 44.2076607
time: 87.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2076608, upper bound: 44.2178111
time: 86.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 173.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 173.91
Output dim: 4, lower bound: -44.2178112, upper bound: 44.2076607
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 173.91
Output dim: 4, lower bound: -44.2076608, upper bound: 44.2178111

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7011414, 79.7020645
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0873184, 106.0868225
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9839249, 81.9832230
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9366760, 85.9361267
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 868

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 533

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2173625, upper bound: 44.1925377
time: 79.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2026773, upper bound: 44.2072119
time: 81.91 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7024841, 79.7011414
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0874100, 106.0873184
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9840164, 81.9839325
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9368591, 85.9366760
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 605

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1464

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2076608, upper bound: 44.2168054
time: 103.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2066580, upper bound: 44.2178111
time: 93.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 199.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 199.14
Output dim: 4, lower bound: -44.2173625, upper bound: 44.1925377
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 199.14
Output dim: 4, lower bound: -44.2026773, upper bound: 44.2072119
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 199.14
Output dim: 4, lower bound: -44.2076608, upper bound: 44.2168054
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 199.14
Output dim: 4, lower bound: -44.2066580, upper bound: 44.2178111

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6959763, 79.6980133
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0711365, 106.0739822
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9595032, 81.9642029
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9209747, 85.9238892
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1604102, upper bound: 44.1903090
time: 109.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2151194, upper bound: 44.1356973
time: 130.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7023010, 79.7007599
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0874100, 106.0873184
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9832001, 81.9832611
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9352798, 85.9353638
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1666

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1631704, upper bound: 44.1723073
time: 85.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1631704, upper bound: 44.2163392
time: 78.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7021179, 79.7009583
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0874100, 106.0873184
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9833374, 81.9831314
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9355240, 85.9351196
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1666

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2062942, upper bound: 44.2053106
time: 82.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1941537, upper bound: 44.2174468
time: 86.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 171.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 171.17
Output dim: 4, lower bound: -44.1604102, upper bound: 44.1903090
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 171.17
Output dim: 4, lower bound: -44.2151194, upper bound: 44.1356973
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 171.17
Output dim: 4, lower bound: -44.1631704, upper bound: 44.1723073
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 171.17
Output dim: 4, lower bound: -44.1631704, upper bound: 44.2163392
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 171.17
Output dim: 4, lower bound: -44.2062942, upper bound: 44.2053106
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 171.17
Output dim: 4, lower bound: -44.1941537, upper bound: 44.2174468

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6923523, 79.6876831
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0966415, 106.0978699
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9906387, 81.9916153
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9354553, 85.9355087
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 853

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 716

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1510234, upper bound: 44.2010553
time: 88.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1479110, upper bound: 44.2041553
time: 76.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7046661, 79.7031708
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0860672, 106.0858765
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9807739, 81.9802856
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9332123, 85.9325333
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 546

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 955

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1929513, upper bound: 44.2168580
time: 79.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1935711, upper bound: 44.2161848
time: 82.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 163.98 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 163.98
Output dim: 4, lower bound: -44.1510234, upper bound: 44.2010553
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 163.98
Output dim: 4, lower bound: -44.1479110, upper bound: 44.2041553
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 163.98
Output dim: 4, lower bound: -44.1929513, upper bound: 44.2168580
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 163.98
Output dim: 4, lower bound: -44.1935711, upper bound: 44.2161848

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6963654, 79.6925964
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0772095, 106.0744247
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9630737, 81.9581985
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9196396, 85.9156265
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 576

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1731

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1602165, upper bound: 44.2155516
time: 95.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1916618, upper bound: 44.1842831
time: 82.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6940765, 79.6948853
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0746155, 106.0770035
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9586945, 81.9625702
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9162979, 85.9189682
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 638

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1245272, upper bound: 44.2142726
time: 128.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1916784, upper bound: 44.1472468
time: 76.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 207.08 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 207.08
Output dim: 4, lower bound: -44.1602165, upper bound: 44.2155516
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 207.08
Output dim: 4, lower bound: -44.1916618, upper bound: 44.1842831
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 207.08
Output dim: 4, lower bound: -44.1245272, upper bound: 44.2142726
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 207.08
Output dim: 4, lower bound: -44.1916784, upper bound: 44.1472468

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.6981125, 79.6947937
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0774841, 106.0739288
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9593430, 81.9540634
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9193268, 85.9152832
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 564

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 766

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1469702, upper bound: 44.2024325
time: 72.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1469702, upper bound: 44.2154489
time: 108.94 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 184.08 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 184.08
Output dim: 4, lower bound: -44.1469702, upper bound: 44.2024325
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 184.08
Output dim: 4, lower bound: -44.1469702, upper bound: 44.2154489

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -56.7373924, 43.5812607, -56.7373924, 43.5812607, -100.3186493, 100.3186493
1: -25.4216919, 37.8586006, -25.4216919, 37.8586006, -63.2802925, 63.2802925
2: -21.9953747, 37.2954292, -21.9953747, 37.2954292, -59.2908020, 59.2908020
3: -24.6426735, 39.9388695, -24.6426735, 39.9388695, -64.5815430, 64.5815430
4: -28.6905251, 43.8665695, -28.6905251, 43.8665695, -72.5570984, 72.5570984
5: -24.8359871, 39.8739700, -24.8359871, 39.8739700, -64.7099609, 64.7099609
6: -54.3298950, 31.9476204, -54.3298950, 31.9476204, -86.2775116, 86.2775116
7: -30.5931587, 39.6691208, -30.5931587, 39.6691208, -70.2622833, 70.2622833
8: -36.7450104, 53.7130432, -36.7450104, 53.7130432, -90.4580536, 90.4580536
9: -29.1745262, 39.1134644, -29.1745262, 39.1134644, -68.2879944, 68.2879944
10: -49.6865082, 43.9997711, -49.6865082, 43.9997711, -93.6862793, 93.6862793
11: -49.2464447, 22.1545029, -49.2464447, 22.1545029, -71.4009476, 71.4009476
12: -55.3999405, 25.4067211, -55.3999405, 25.4067211, -79.7009125, 79.6979675
13: -50.7466698, 43.8802795, -50.7466698, 43.8802795, -94.6269531, 94.6269531
14: -87.6760712, 31.4053574, -87.6760712, 31.4053574, -119.0814285, 119.0814285
15: -35.9532700, 36.1003151, -35.9532700, 36.1003151, -72.0535889, 72.0535889
16: -46.0161438, 34.0737915, -46.0161438, 34.0737915, -80.0899353, 80.0899353
17: -85.0268250, 23.6481457, -85.0268250, 23.6481457, -108.6749725, 108.6749725
18: -49.1281204, 31.5272331, -49.1281204, 31.5272331, -80.6553497, 80.6553497
19: -39.0521889, 18.6601238, -39.0521889, 18.6601238, -57.7123108, 57.7123108
20: -37.0207062, 23.4827690, -37.0207062, 23.4827690, -60.5034752, 60.5034752
21: -48.2837219, 22.2476826, -48.2837219, 22.2476826, -70.5314026, 70.5314026
22: -50.1172447, 22.2121162, -50.1172447, 22.2121162, -72.3293610, 72.3293610
23: -39.2036057, 24.1490345, -39.2036057, 24.1490345, -63.3526382, 63.3526382
24: -46.4198875, 24.1115894, -46.4198875, 24.1115894, -70.5314789, 70.5314789
25: -41.3342056, 24.9019737, -41.3342056, 24.9019737, -66.2361755, 66.2361755
26: -57.1147995, 33.7629852, -57.1147995, 33.7629852, -90.8777847, 90.8777847
27: -45.3962479, 28.8366470, -45.3962479, 28.8366470, -74.2328949, 74.2328949
28: -39.0445175, 26.8544044, -39.0445175, 26.8544044, -65.8989258, 65.8989258
29: -51.7799759, 20.7330284, -51.7799759, 20.7330284, -72.5130005, 72.5130005
30: -49.4135399, 26.3465347, -49.4135399, 26.3465347, -75.7600708, 75.7600708
31: -51.1895180, 28.0085526, -51.1895180, 28.0085526, -79.1980743, 79.1980743
32: -52.4488831, 24.7276020, -52.4488831, 24.7276020, -77.1764832, 77.1764832
33: -72.4477234, 33.8769150, -72.4477234, 33.8769150, -106.0751801, 106.0708771
34: -65.5867767, 17.2007103, -65.5867767, 17.2007103, -81.9568024, 81.9506454
35: -63.8561935, 23.5946140, -63.8561935, 23.5946140, -85.9187317, 85.9144745
36: -62.0380096, 24.4423981, -62.0380096, 24.4423981, -86.4804077, 86.4804077
37: -87.2119446, 19.9378090, -87.2119446, 19.9378090, -107.1497498, 107.1497498
38: -70.0268326, 29.2647362, -70.0268326, 29.2647362, -99.2915649, 99.2915649
39: -80.5317535, 30.6962433, -80.5317535, 30.6962433, -111.2279968, 111.2279968
40: -62.6256981, 25.7695351, -62.6256981, 25.7695351, -88.3952332, 88.3952332
41: -55.0449638, 32.9426651, -55.0449638, 32.9426651, -87.9876251, 87.9876251
42: -36.2887535, 26.0834236, -36.2887535, 26.0834236, -62.3721771, 62.3721771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=229, inp2_unstable=229, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1702

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 686

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1464089, upper bound: 44.2123408
time: 75.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1438087, upper bound: 44.2148844
time: 87.86 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 165.58 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 165.58
Output dim: 4, lower bound: -44.1464089, upper bound: 44.2123408
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 165.58
Output dim: 4, lower bound: -44.1438087, upper bound: 44.2148844

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 104.39 + 2184.20 = 2288.58 seconds
