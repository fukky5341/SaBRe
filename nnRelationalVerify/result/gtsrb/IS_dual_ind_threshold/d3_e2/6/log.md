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
execution time: IAR + RelationalAnalysis = 2.95 + 98.14 = 101.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -44.2596780, upper bound: 44.2596780

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1607572, upper bound: 44.2526825
time: 88.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1607572, upper bound: 44.2554161
time: 96.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 184.54 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 184.54
Output dim: 4, lower bound: -44.1607572, upper bound: 44.2526825
IS_A2, status: Status.UNKNOWN, split count: 1, time: 184.54
Output dim: 4, lower bound: -44.1607572, upper bound: 44.2554161

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -56.5105743, 43.5065536, -56.6740112, 43.5609665, -100.0715408, 100.1805649
1: -25.2357826, 37.7437363, -25.3654137, 37.8443069, -63.0800896, 63.1091499
2: -21.7595882, 37.1718941, -21.9197216, 37.2803383, -59.0399246, 59.0916138
3: -24.3506813, 39.7547417, -24.5467014, 39.9178696, -64.2685547, 64.3014450
4: -28.4041519, 43.7163277, -28.5995274, 43.8512001, -72.2553558, 72.3158569
5: -24.5815754, 39.7380600, -24.7549191, 39.8543701, -64.4359436, 64.4929810
6: -54.2311058, 31.8290653, -54.3021851, 31.9133530, -86.1444550, 86.1312485
7: -30.3759174, 39.5928993, -30.5285721, 39.6540222, -70.0299377, 70.1214752
8: -36.5223160, 53.5420761, -36.6746254, 53.6893730, -90.2116852, 90.2167053
9: -29.0552959, 39.0042038, -29.1408424, 39.0836258, -68.1389236, 68.1450500
10: -49.3937454, 43.5561371, -49.6485062, 43.8555565, -93.2492981, 93.2046432
11: -48.9253693, 21.7653618, -49.2147675, 22.0214233, -70.9467926, 70.9801331
12: -55.2292557, 25.0378342, -55.3773956, 25.2863045, -79.4080734, 79.3042145
13: -50.4697456, 43.6687889, -50.6561127, 43.8454323, -94.3151779, 94.3249054
14: -87.3682709, 30.9267197, -87.6310883, 31.2431698, -118.6114426, 118.5578079
15: -35.6922836, 35.9926987, -35.8719330, 36.0736237, -71.7659073, 71.8646317
16: -45.7893143, 33.8073273, -45.9635849, 33.9924316, -79.7817459, 79.7709122
17: -84.7682266, 23.3209648, -84.9995270, 23.5405254, -108.3087540, 108.3204956
18: -48.8703384, 31.2241402, -49.1016922, 31.4241257, -80.2944641, 80.3258362
19: -38.8239517, 18.4238987, -39.0217743, 18.5812855, -57.4052353, 57.4456711
20: -36.8425255, 23.2638512, -36.9910660, 23.4103565, -60.2528839, 60.2549171
21: -48.0176277, 21.9491444, -48.2512627, 22.1480827, -70.1657104, 70.2004089
22: -49.9893456, 22.0663948, -50.0885620, 22.1665745, -72.1559219, 72.1549530
23: -39.0074539, 23.8845139, -39.1774673, 24.0626335, -63.0700874, 63.0619812
24: -46.2343788, 23.9189415, -46.3900833, 24.0471458, -70.2815247, 70.3090210
25: -41.1693687, 24.6877842, -41.3060760, 24.8315926, -66.0009613, 65.9938583
26: -56.8998337, 33.3579407, -57.0829659, 33.6317062, -90.5315399, 90.4409027
27: -45.2530785, 28.7227898, -45.3623199, 28.7989655, -74.0520477, 74.0851135
28: -38.8920593, 26.6516533, -39.0161552, 26.7887993, -65.6808624, 65.6678085
29: -51.6282768, 20.5402222, -51.7550354, 20.6702480, -72.2985229, 72.2952576
30: -49.2070541, 26.0297756, -49.3852768, 26.2431297, -75.4501801, 75.4150543
31: -50.8971748, 27.7076969, -51.1518555, 27.9069672, -78.8041382, 78.8595505
32: -52.3358345, 24.6303749, -52.4184341, 24.7001190, -77.0359497, 77.0488129
33: -72.1747742, 33.6907043, -72.3608093, 33.8523026, -105.7750626, 105.8329697
34: -65.4369507, 17.0753345, -65.5424805, 17.1703968, -81.7703247, 81.8217926
35: -63.6814270, 23.4546967, -63.8013802, 23.5747547, -85.7114868, 85.7199020
36: -61.8688507, 24.3281269, -61.9847908, 24.4238968, -86.2927475, 86.3129196
37: -87.0766754, 19.8058987, -87.1753159, 19.8997784, -106.9764557, 106.9812164
38: -69.7985916, 29.1040726, -69.9575882, 29.2396679, -99.0382614, 99.0616608
39: -80.2962189, 30.5586891, -80.4578629, 30.6773949, -110.9736176, 111.0165558
40: -62.4894905, 25.6665001, -62.5884819, 25.7481384, -88.2376251, 88.2549820
41: -54.9029922, 32.8344803, -55.0038605, 32.9150581, -87.8180542, 87.8383408
42: -36.2115707, 25.9148102, -36.2681885, 26.0376358, -62.2492065, 62.1829987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2469635
time: 84.91 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2469635
time: 91.20 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -56.7224731, 43.5684662, -56.7324448, 43.5771065, -100.2995758, 100.3009109
1: -25.4107361, 37.8542671, -25.4180183, 37.8571854, -63.2679214, 63.2722855
2: -21.9808788, 37.2913666, -21.9904747, 37.2940063, -59.2748871, 59.2818413
3: -24.6240921, 39.9338074, -24.6365089, 39.9370728, -64.5611649, 64.5703125
4: -28.6741638, 43.8617554, -28.6848812, 43.8649292, -72.5390930, 72.5466385
5: -24.8202057, 39.8691750, -24.8307686, 39.8722305, -64.6924362, 64.6999435
6: -54.3231697, 31.9093189, -54.3275757, 31.9350319, -86.2582016, 86.2368927
7: -30.5801163, 39.6625938, -30.5887451, 39.6669693, -70.2470856, 70.2513428
8: -36.7310638, 53.7063789, -36.7402802, 53.7107544, -90.4418182, 90.4466553
9: -29.1671181, 39.0983047, -29.1719284, 39.1085205, -68.2756348, 68.2702332
10: -49.6786575, 43.9714355, -49.6837997, 43.9902611, -93.6689148, 93.6552353
11: -49.2375870, 22.1375771, -49.2432938, 22.1487274, -71.3863144, 71.3808746
12: -55.3928757, 25.3833237, -55.3975830, 25.3987350, -79.6872025, 79.6579285
13: -50.7037506, 43.8716240, -50.7311440, 43.8773460, -94.5811005, 94.6027679
14: -87.6623840, 31.3777580, -87.6715927, 31.3961048, -119.0584869, 119.0493469
15: -35.9035568, 36.0934067, -35.9363785, 36.0978928, -72.0014496, 72.0297852
16: -46.0032501, 33.9972534, -46.0115891, 34.0485802, -80.0518341, 80.0088425
17: -85.0176239, 23.6278248, -85.0237274, 23.6412449, -108.6588669, 108.6515503
18: -49.1204071, 31.5075302, -49.1255302, 31.5208035, -80.6412125, 80.6330566
19: -39.0456619, 18.6455898, -39.0498962, 18.6553783, -57.7010422, 57.6954880
20: -37.0133743, 23.4696064, -37.0182343, 23.4784012, -60.4917755, 60.4878387
21: -48.2751007, 22.2303524, -48.2808075, 22.2419949, -70.5170975, 70.5111618
22: -50.1047325, 22.2024040, -50.1131325, 22.2087917, -72.3135223, 72.3155365
23: -39.1971054, 24.1337147, -39.2013931, 24.1439476, -63.3410530, 63.3351059
24: -46.4095917, 24.0987396, -46.4162140, 24.1074829, -70.5170746, 70.5149536
25: -41.3257599, 24.8886948, -41.3311462, 24.8975544, -66.2233124, 66.2198410
26: -57.1041832, 33.7392120, -57.1111755, 33.7551193, -90.8592987, 90.8503876
27: -45.3861351, 28.8112659, -45.3927612, 28.8276749, -74.2138062, 74.2040253
28: -39.0380058, 26.8418293, -39.0422668, 26.8502350, -65.8882446, 65.8840942
29: -51.7683792, 20.7213688, -51.7761269, 20.7290192, -72.4973984, 72.4974976
30: -49.4046173, 26.3280239, -49.4103546, 26.3398170, -75.7444305, 75.7383804
31: -51.1814537, 27.9898815, -51.1868134, 28.0025768, -79.1840286, 79.1766968
32: -52.4336624, 24.7147255, -52.4438057, 24.7231560, -77.1568146, 77.1585312
33: -72.4337540, 33.8702202, -72.4428024, 33.8746338, -106.0392914, 106.0749435
34: -65.5638962, 17.1934624, -65.5794144, 17.1982021, -81.9464722, 81.9716339
35: -63.8295860, 23.5901337, -63.8470917, 23.5930882, -85.9089355, 85.9268646
36: -62.0198212, 24.4382763, -62.0318718, 24.4409771, -86.4608002, 86.4701462
37: -87.2017288, 19.9145813, -87.2082062, 19.9297409, -107.1314697, 107.1227875
38: -70.0081940, 29.2580490, -70.0205536, 29.2624569, -99.2706528, 99.2786026
39: -80.5160828, 30.6909485, -80.5260162, 30.6944828, -111.2105637, 111.2169647
40: -62.6156311, 25.7414398, -62.6222000, 25.7605896, -88.3762207, 88.3636398
41: -55.0349960, 32.9260941, -55.0416107, 32.9373131, -87.9723053, 87.9677048
42: -36.2835846, 26.0485916, -36.2869759, 26.0722904, -62.3558731, 62.3355675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2485271
time: 87.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2485271
time: 87.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 178.15 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 178.15
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2469635
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 178.15
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2469635
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 178.15
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2485271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 178.15
Output dim: 4, lower bound: -44.1159836, upper bound: 44.2485271

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -56.4953575, 43.4775887, -56.5146713, 43.4539032, -99.9492645, 99.9922638
1: -25.2268944, 37.7124252, -25.2571526, 37.7393913, -62.9662857, 62.9695778
2: -21.7496681, 37.1492348, -21.7943306, 37.2037697, -58.9534378, 58.9435654
3: -24.3454094, 39.7417755, -24.4883270, 39.8557968, -64.2012024, 64.2301025
4: -28.3910713, 43.6678391, -28.4152164, 43.6976166, -72.0886841, 72.0830536
5: -24.5736732, 39.7080498, -24.6475410, 39.7394409, -64.3131104, 64.3555908
6: -54.2234688, 31.8058586, -54.2325211, 31.8140106, -86.0374756, 86.0383759
7: -30.3634777, 39.5513840, -30.3755417, 39.5098572, -69.8733368, 69.9269257
8: -36.5141869, 53.5092049, -36.5578270, 53.5711594, -90.0853424, 90.0670319
9: -29.0475292, 38.9880981, -29.0805988, 39.0049896, -68.0525208, 68.0686951
10: -49.3564453, 43.5439911, -49.5067291, 43.6756744, -93.0321198, 93.0507202
11: -48.8819046, 21.7572918, -49.0585861, 21.8869133, -70.7688141, 70.8158798
12: -55.1791992, 25.0289803, -55.2002640, 25.1035805, -79.1697769, 79.1112823
13: -50.4580116, 43.6557388, -50.5758133, 43.7673416, -94.2253571, 94.2315521
14: -87.2883224, 30.9196758, -87.3377838, 30.9674435, -118.2557678, 118.2574615
15: -35.6672935, 35.9841690, -35.7482681, 36.0210571, -71.6883545, 71.7324371
16: -45.7745056, 33.7656708, -45.8480339, 33.8262177, -79.6007233, 79.6137085
17: -84.6933594, 23.3119888, -84.7336273, 23.3526688, -108.0460281, 108.0456161
18: -48.8525620, 31.2132397, -49.0561905, 31.3720169, -80.2245789, 80.2694321
19: -38.8089256, 18.4213676, -38.9441452, 18.5481091, -57.3570328, 57.3655128
20: -36.8239288, 23.2583084, -36.9049683, 23.3249340, -60.1488647, 60.1632767
21: -47.9907303, 21.9440002, -48.1346855, 22.0670319, -70.0577621, 70.0786896
22: -49.9318085, 22.0587139, -49.8814926, 22.0382786, -71.9700851, 71.9402084
23: -38.9737167, 23.8793716, -39.0493774, 23.9510956, -62.9248123, 62.9287491
24: -46.2053604, 23.9144840, -46.2673378, 23.9645119, -70.1698761, 70.1818237
25: -41.1270905, 24.6757851, -41.1523132, 24.6787968, -65.8058853, 65.8280945
26: -56.8584328, 33.3520737, -56.9155922, 33.4884872, -90.3469238, 90.2676697
27: -45.2413712, 28.7112694, -45.2869110, 28.7432213, -73.9845886, 73.9981842
28: -38.8680687, 26.6460876, -38.9124489, 26.6863556, -65.5544281, 65.5585327
29: -51.5615692, 20.5331268, -51.5191994, 20.5069637, -72.0685349, 72.0523224
30: -49.1665115, 26.0198650, -49.2455826, 26.0863037, -75.2528152, 75.2654495
31: -50.8762779, 27.7028885, -51.0572395, 27.8372993, -78.7135773, 78.7601318
32: -52.3191872, 24.6242599, -52.3336220, 24.6522045, -76.9713898, 76.9578857
33: -72.1625137, 33.6648941, -72.2326508, 33.7508316, -105.6664581, 105.6114655
34: -65.4216690, 17.0649185, -65.4628677, 17.1041584, -81.7334366, 81.7059784
35: -63.6700363, 23.4459324, -63.7279358, 23.5340004, -85.6730804, 85.6004944
36: -61.8581696, 24.3174515, -61.9083214, 24.3688335, -86.2270050, 86.2257690
37: -87.0601501, 19.7667332, -87.0247803, 19.7556744, -106.8158264, 106.7915115
38: -69.7859039, 29.0689716, -69.8436127, 29.1161118, -98.9020157, 98.9125824
39: -80.2821350, 30.5153809, -80.2977142, 30.5322285, -110.8143616, 110.8130951
40: -62.4777374, 25.6079674, -62.4527168, 25.5599976, -88.0377350, 88.0606842
41: -54.8932571, 32.8007889, -54.8770790, 32.7935181, -87.6867752, 87.6778717
42: -36.2022476, 25.9005966, -36.2191925, 25.9737282, -62.1759758, 62.1197891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2376852
time: 90.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2415128
time: 90.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -56.5083122, 43.5028954, -56.6672440, 43.5512276, -100.0595398, 100.1701355
1: -25.2343521, 37.7416954, -25.3610325, 37.8384933, -63.0728455, 63.1027298
2: -21.7579899, 37.1704597, -21.9149628, 37.2756081, -59.0335999, 59.0854225
3: -24.3493271, 39.7534790, -24.5427303, 39.9142303, -64.2635574, 64.2962112
4: -28.4024315, 43.7138596, -28.5943832, 43.8410797, -72.2435150, 72.3082428
5: -24.5800571, 39.7363510, -24.7503548, 39.8492622, -64.4293213, 64.4867096
6: -54.2298737, 31.8249226, -54.2984848, 31.9019032, -86.1317749, 86.1234055
7: -30.3738651, 39.5899773, -30.5223808, 39.6460495, -70.0199127, 70.1123581
8: -36.5206833, 53.5399170, -36.6697731, 53.6829834, -90.2036667, 90.2096863
9: -29.0541382, 38.9975815, -29.1370773, 39.0636978, -68.1178360, 68.1346588
10: -49.3913193, 43.5535355, -49.6408691, 43.8476791, -93.2389984, 93.1944046
11: -48.9226913, 21.7638912, -49.2038116, 22.0174770, -70.9401703, 70.9677048
12: -55.2262726, 25.0351372, -55.3689232, 25.2783356, -79.3970795, 79.2788849
13: -50.4665604, 43.6660233, -50.6466484, 43.8378220, -94.3043823, 94.3126678
14: -87.3634720, 30.9244537, -87.6173401, 31.2363377, -118.5998077, 118.5417938
15: -35.6865654, 35.9913712, -35.8535194, 36.0694504, -71.7560120, 71.8448944
16: -45.7869453, 33.7952423, -45.9561615, 33.9542084, -79.7411499, 79.7514038
17: -84.7639618, 23.3188515, -84.9877014, 23.5341225, -108.2980804, 108.3065491
18: -48.8682785, 31.2084980, -49.0960693, 31.3706055, -80.2388840, 80.3045654
19: -38.8225937, 18.4232731, -39.0176392, 18.5794525, -57.4020462, 57.4409103
20: -36.8410645, 23.2629700, -36.9862099, 23.4079018, -60.2489662, 60.2491798
21: -48.0157127, 21.9483166, -48.2453995, 22.1457443, -70.1614532, 70.1937180
22: -49.9847870, 22.0648212, -50.0761147, 22.1618176, -72.1466064, 72.1409378
23: -39.0054245, 23.8836594, -39.1715317, 24.0601215, -63.0655441, 63.0551910
24: -46.2319908, 23.9182816, -46.3832245, 24.0453720, -70.2773590, 70.3015060
25: -41.1668091, 24.6864700, -41.2985840, 24.8277550, -65.9945679, 65.9850540
26: -56.8964767, 33.3557663, -57.0739899, 33.6247406, -90.5212173, 90.4297562
27: -45.2510986, 28.7183475, -45.3566589, 28.7843742, -74.0354767, 74.0750046
28: -38.8899345, 26.6506920, -39.0087433, 26.7859955, -65.6759338, 65.6594391
29: -51.6236305, 20.5389671, -51.7417107, 20.6665039, -72.2901306, 72.2806778
30: -49.2046127, 26.0280075, -49.3758430, 26.2385216, -75.4431305, 75.4038544
31: -50.8952751, 27.7068787, -51.1461449, 27.9046059, -78.7998810, 78.8530273
32: -52.3331757, 24.6292953, -52.4099426, 24.6971340, -77.0303116, 77.0392380
33: -72.1731110, 33.6889038, -72.3560028, 33.8453102, -105.7459717, 105.8873138
34: -65.4303436, 17.0739479, -65.5217209, 17.1660557, -81.7295303, 81.8568878
35: -63.6783905, 23.4530964, -63.7922554, 23.5707722, -85.6931534, 85.7641296
36: -61.8669739, 24.3230782, -61.9798393, 24.4112110, -86.2781830, 86.3029175
37: -87.0746460, 19.7898769, -87.1694183, 19.8654346, -106.9400787, 106.9592972
38: -69.7963028, 29.0999374, -69.9513168, 29.2324753, -99.0287781, 99.0512543
39: -80.2938995, 30.5550499, -80.4510651, 30.6648502, -110.9587479, 111.0061188
40: -62.4874763, 25.6604671, -62.5825386, 25.7358322, -88.2233124, 88.2430038
41: -54.9014511, 32.8292580, -54.9992180, 32.9043503, -87.8058014, 87.8284760
42: -36.2029953, 25.9130096, -36.2424316, 26.0328388, -62.2358322, 62.1554413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2378431
time: 80.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2415128
time: 90.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -56.7070312, 43.5391617, -56.5725136, 43.4698753, -100.1769104, 100.1116791
1: -25.4019203, 37.8229599, -25.3091335, 37.7522202, -63.1541405, 63.1320953
2: -21.9709244, 37.2695007, -21.8644676, 37.2173462, -59.1882706, 59.1339684
3: -24.6187611, 39.9205055, -24.5764084, 39.8748703, -64.4936295, 64.4969177
4: -28.6607227, 43.8151054, -28.5012951, 43.7111588, -72.3718796, 72.3163986
5: -24.8122139, 39.8389130, -24.7220001, 39.7571182, -64.5693359, 64.5609131
6: -54.3152847, 31.8856487, -54.2570114, 31.8344536, -86.1497345, 86.1426620
7: -30.5676422, 39.6211700, -30.4347687, 39.5230331, -70.0906754, 70.0559387
8: -36.7229271, 53.6733932, -36.6226196, 53.5924644, -90.3153915, 90.2960129
9: -29.1590729, 39.0821533, -29.1107044, 39.0292130, -68.1882858, 68.1928558
10: -49.6415176, 43.9595108, -49.5416183, 43.8071594, -93.4486771, 93.5011292
11: -49.1950951, 22.1293373, -49.0870056, 22.0118408, -71.2069397, 71.2163391
12: -55.3428040, 25.3744087, -55.2202988, 25.2137146, -79.4462738, 79.4707489
13: -50.6922607, 43.8577728, -50.6500931, 43.7986603, -94.4909210, 94.5078659
14: -87.5818481, 31.3704948, -87.3778839, 31.1167068, -118.6985550, 118.7483826
15: -35.8806419, 36.0848389, -35.8144302, 36.0448608, -71.9255066, 71.8992691
16: -45.9883652, 33.9575539, -45.8954315, 33.8825836, -79.8709488, 79.8529816
17: -84.9421997, 23.6187401, -84.7576447, 23.4515190, -108.3937225, 108.3763885
18: -49.1013451, 31.5010986, -49.0796585, 31.4677353, -80.5690765, 80.5807571
19: -39.0305557, 18.6431980, -38.9719696, 18.6213264, -57.6518822, 57.6151657
20: -36.9951248, 23.4640102, -36.9318390, 23.3919811, -60.3871078, 60.3958511
21: -48.2481689, 22.2252083, -48.1637878, 22.1596603, -70.4078293, 70.3889923
22: -50.0471916, 22.1947002, -49.9061012, 22.0790520, -72.1262436, 72.1007996
23: -39.1633148, 24.1285286, -39.0729866, 24.0307026, -63.1940155, 63.2015152
24: -46.3802452, 24.0941257, -46.2933998, 24.0237484, -70.4039917, 70.3875275
25: -41.2835350, 24.8763866, -41.1774979, 24.7433624, -66.0269012, 66.0538864
26: -57.0624809, 33.7334518, -56.9438858, 33.6103859, -90.6728668, 90.6773376
27: -45.3740005, 28.8024158, -45.3164368, 28.7742271, -74.1482239, 74.1188507
28: -39.0142708, 26.8362637, -38.9381866, 26.7467995, -65.7610703, 65.7744522
29: -51.7017975, 20.7139931, -51.5401993, 20.5645771, -72.2663727, 72.2541962
30: -49.3655701, 26.3179035, -49.2706375, 26.1855583, -75.5511322, 75.5885391
31: -51.1604195, 27.9850616, -51.0916519, 27.9316959, -79.0921173, 79.0767136
32: -52.4177933, 24.7085781, -52.3596687, 24.6750069, -77.0928040, 77.0682449
33: -72.4211884, 33.8450470, -72.3133850, 33.7729187, -105.9484863, 105.8516922
34: -65.5484161, 17.1832142, -65.4987030, 17.1314392, -81.9076157, 81.8540955
35: -63.8183975, 23.5810261, -63.7724075, 23.5521164, -85.8735199, 85.7938843
36: -62.0092201, 24.4265499, -61.9505806, 24.3857765, -86.3949966, 86.3771286
37: -87.1846848, 19.8743458, -87.0563812, 19.7856350, -106.9703217, 106.9307251
38: -69.9956131, 29.2229328, -69.9035797, 29.1386166, -99.1342316, 99.1265106
39: -80.5021133, 30.6473656, -80.3626175, 30.5492020, -111.0513153, 111.0099792
40: -62.6032715, 25.6836777, -62.4847794, 25.5706635, -88.1739349, 88.1684570
41: -55.0250854, 32.8920860, -54.9107552, 32.8154030, -87.8404846, 87.8028412
42: -36.2738419, 26.0334969, -36.2376251, 26.0067215, -62.2805634, 62.2711220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2390322
time: 104.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2428141
time: 75.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -56.7199211, 43.5641479, -56.7251587, 43.5666656, -100.2865906, 100.2893066
1: -25.4090443, 37.8521233, -25.4131241, 37.8510361, -63.2600784, 63.2652473
2: -21.9790249, 37.2893944, -21.9851418, 37.2890778, -59.2681046, 59.2745361
3: -24.6225739, 39.9324760, -24.6320992, 39.9332237, -64.5558014, 64.5645752
4: -28.6721535, 43.8575516, -28.6791172, 43.8546677, -72.5268250, 72.5366669
5: -24.8184853, 39.8673859, -24.8257656, 39.8669662, -64.6854553, 64.6931534
6: -54.3218307, 31.9049244, -54.3237152, 31.9226875, -86.2445221, 86.2286377
7: -30.5778503, 39.6594620, -30.5820808, 39.6583557, -70.2362061, 70.2415466
8: -36.7292061, 53.7040253, -36.7348824, 53.7039948, -90.4331970, 90.4389038
9: -29.1657524, 39.0915833, -29.1679268, 39.0881882, -68.2539368, 68.2595062
10: -49.6760178, 43.9682159, -49.6759415, 43.9812622, -93.6572800, 93.6441574
11: -49.2338257, 22.1358109, -49.2320557, 22.1440163, -71.3778381, 71.3678665
12: -55.3897209, 25.3804703, -55.3886032, 25.3903370, -79.6756287, 79.6322021
13: -50.7001419, 43.8690567, -50.7211151, 43.8696060, -94.5697479, 94.5901718
14: -87.6573486, 31.3754578, -87.6571350, 31.3891621, -119.0465088, 119.0325928
15: -35.8953514, 36.0918770, -35.9143448, 36.0935211, -71.9888763, 72.0062256
16: -46.0005493, 33.9831505, -46.0038376, 34.0086517, -80.0092010, 79.9869843
17: -85.0133209, 23.6254826, -85.0112915, 23.6345863, -108.6479034, 108.6367722
18: -49.1182976, 31.4876347, -49.1195641, 31.4625969, -80.5808945, 80.6072006
19: -39.0441704, 18.6447010, -39.0455513, 18.6530247, -57.6971970, 57.6902542
20: -37.0114746, 23.4686508, -37.0131378, 23.4756832, -60.4871597, 60.4817886
21: -48.2729759, 22.2293396, -48.2746811, 22.2392082, -70.5121841, 70.5040207
22: -50.0997314, 22.2007065, -50.0996323, 22.2037888, -72.3035202, 72.3003387
23: -39.1950455, 24.1326771, -39.1952820, 24.1410103, -63.3360558, 63.3279572
24: -46.4070129, 24.0980358, -46.4090157, 24.1055222, -70.5125351, 70.5070496
25: -41.3230133, 24.8872757, -41.3233032, 24.8934631, -66.2164764, 66.2105789
26: -57.1006622, 33.7367210, -57.1014595, 33.7478561, -90.8485184, 90.8381805
27: -45.3839417, 28.8058014, -45.3867264, 28.8128834, -74.1968231, 74.1925278
28: -39.0354614, 26.8406677, -39.0346832, 26.8469200, -65.8823853, 65.8753510
29: -51.7632141, 20.7199898, -51.7619095, 20.7249756, -72.4881897, 72.4819031
30: -49.4007187, 26.3259659, -49.4006805, 26.3344212, -75.7351379, 75.7266464
31: -51.1794472, 27.9888687, -51.1808510, 27.9997959, -79.1792450, 79.1697235
32: -52.4294739, 24.7135220, -52.4338951, 24.7198238, -77.1492996, 77.1474152
33: -72.4319611, 33.8677216, -72.4376907, 33.8672943, -106.0115738, 106.1233673
34: -65.5560455, 17.1918278, -65.5573883, 17.1934948, -81.8994675, 82.0157928
35: -63.8261719, 23.5887356, -63.8377609, 23.5889874, -85.8873978, 85.9734039
36: -62.0177078, 24.4341125, -62.0262566, 24.4282551, -86.4459610, 86.4603729
37: -87.1995163, 19.9005795, -87.2019958, 19.8942451, -107.0937653, 107.1025772
38: -70.0054474, 29.2554359, -70.0131836, 29.2549477, -99.2603912, 99.2686157
39: -80.5134277, 30.6864052, -80.5186234, 30.6816616, -111.1950912, 111.2050323
40: -62.6134224, 25.7326889, -62.6159821, 25.7441559, -88.3575745, 88.3486710
41: -55.0333786, 32.9210625, -55.0368004, 32.9251480, -87.9585266, 87.9578629
42: -36.2751160, 26.0466747, -36.2610855, 26.0669403, -62.3420563, 62.3077621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2390322
time: 73.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2428141
time: 85.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 161.55 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2376852
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2415128
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2378431
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2415128
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2390322
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2428141
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2390322
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 161.55
Output dim: 4, lower bound: -44.0250885, upper bound: 44.2428141

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -56.2936707, 43.3947563, -56.4551888, 43.4382935, -99.7319641, 99.8499451
1: -25.0757961, 37.6159363, -25.2107353, 37.7288513, -62.8046494, 62.8266716
2: -21.5238514, 37.0258636, -21.7209530, 37.1922646, -58.7161179, 58.7468185
3: -24.0877991, 39.5968132, -24.4043732, 39.8418198, -63.9296188, 64.0011902
4: -28.1439819, 43.5426788, -28.3351326, 43.6858826, -71.8298645, 71.8778076
5: -24.3356400, 39.5811768, -24.5711727, 39.7257996, -64.0614395, 64.1523514
6: -54.1356812, 31.6780357, -54.2121887, 31.7756023, -85.9112854, 85.8902283
7: -30.1656952, 39.4789543, -30.3156395, 39.4988556, -69.6645508, 69.7945938
8: -36.3157959, 53.3700638, -36.4947739, 53.5545349, -89.8703308, 89.8648376
9: -28.9398880, 38.9083862, -29.0473175, 38.9841461, -67.9240341, 67.9557037
10: -49.2183685, 43.3034744, -49.4784317, 43.6025238, -92.8208923, 92.7819061
11: -48.6418839, 21.4513969, -49.0341072, 21.7841587, -70.4260406, 70.4855042
12: -55.0230331, 24.6989479, -55.1833801, 24.9962788, -78.9030533, 78.7605591
13: -50.2624664, 43.5082626, -50.5128174, 43.7379837, -94.0004501, 94.0210800
14: -87.0734253, 30.5673523, -87.3036270, 30.8492603, -117.9226837, 117.8709793
15: -35.4648666, 35.8999939, -35.6838570, 36.0015869, -71.4664536, 71.5838470
16: -45.6242752, 33.5763969, -45.8092804, 33.7696152, -79.3938904, 79.3856812
17: -84.4765625, 23.0375862, -84.7111969, 23.2616806, -107.7382431, 107.7487793
18: -48.6576157, 30.9831924, -49.0371933, 31.2948799, -79.9524994, 80.0203857
19: -38.6356125, 18.2228107, -38.9218369, 18.4820290, -57.1176414, 57.1446457
20: -36.6771431, 23.0754013, -36.8810959, 23.2642937, -59.9414368, 59.9564972
21: -47.7794189, 21.6953087, -48.1094322, 21.9837379, -69.7631531, 69.8047409
22: -49.8083420, 21.9049683, -49.8593597, 21.9893188, -71.7976608, 71.7643280
23: -38.8273163, 23.6760197, -39.0300598, 23.8843231, -62.7116394, 62.7060776
24: -46.0790443, 23.7698002, -46.2461853, 23.9167843, -69.9958267, 70.0159836
25: -41.0177689, 24.5066490, -41.1323166, 24.6231117, -65.6408844, 65.6389618
26: -56.6529388, 32.9931107, -56.8895264, 33.3697052, -90.0226440, 89.8826370
27: -45.0814323, 28.5819225, -45.2615547, 28.6963406, -73.7777710, 73.8434753
28: -38.7176285, 26.4388123, -38.8902130, 26.6179371, -65.3355637, 65.3290253
29: -51.4186440, 20.3348961, -51.4998131, 20.4415531, -71.8601990, 71.8347092
30: -49.0034637, 25.7646351, -49.2244339, 26.0028152, -75.0062790, 74.9890671
31: -50.6753960, 27.4865284, -51.0309105, 27.7650852, -78.4404831, 78.5174408
32: -52.2185402, 24.5067673, -52.3104935, 24.6167831, -76.8353271, 76.8172607
33: -71.9551086, 33.5179214, -72.1662750, 33.7265358, -105.4218979, 105.3860550
34: -65.3283234, 16.9577618, -65.4369659, 17.0763092, -81.5899811, 81.5493774
35: -63.5799713, 23.3730392, -63.7016144, 23.5185051, -85.5502625, 85.4844208
36: -61.7841110, 24.2449703, -61.8900185, 24.3502750, -86.1343842, 86.1349869
37: -86.9517059, 19.6229515, -87.0013504, 19.7110157, -106.6627197, 106.6242981
38: -69.6623993, 28.9878521, -69.8102264, 29.0972443, -98.7596436, 98.7980804
39: -80.1196518, 30.4121952, -80.2473221, 30.5156021, -110.6352539, 110.6595154
40: -62.3729782, 25.5115128, -62.4281425, 25.5378075, -87.9107819, 87.9396515
41: -54.8057251, 32.6812782, -54.8538933, 32.7597427, -87.5654678, 87.5351715
42: -36.1302795, 25.7606659, -36.2021141, 25.9331856, -62.0634651, 61.9627800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2301780
time: 73.16 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2338611
time: 98.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -56.4825478, 43.4727707, -56.5107231, 43.4523544, -99.9349060, 99.9834900
1: -25.2166405, 37.7078476, -25.2540226, 37.7379227, -62.9545631, 62.9618683
2: -21.7346001, 37.1449165, -21.7897263, 37.2024040, -58.9370041, 58.9346428
3: -24.3284225, 39.7364464, -24.4830761, 39.8541107, -64.1825333, 64.2195206
4: -28.3742294, 43.6639595, -28.4100399, 43.6963730, -72.0706024, 72.0739975
5: -24.5581684, 39.7036362, -24.6427803, 39.7380295, -64.2961960, 64.3464203
6: -54.2174187, 31.7681198, -54.2306328, 31.8022575, -86.0196762, 85.9987488
7: -30.3495598, 39.5446167, -30.3712196, 39.5076790, -69.8572388, 69.9158325
8: -36.5008240, 53.5025711, -36.5537720, 53.5690613, -90.0698853, 90.0563431
9: -29.0361176, 38.9802742, -29.0770626, 39.0026360, -68.0387573, 68.0573349
10: -49.3486557, 43.5205154, -49.5043144, 43.6684647, -93.0171204, 93.0248260
11: -48.8722534, 21.7356873, -49.0555115, 21.8801022, -70.7523575, 70.7911987
12: -55.1724434, 25.0059280, -55.1981354, 25.0963593, -79.1556473, 79.0615692
13: -50.4027023, 43.6474953, -50.5581093, 43.7647591, -94.1674652, 94.2056046
14: -87.2769089, 30.8958130, -87.3341293, 30.9600124, -118.2369232, 118.2299423
15: -35.6114693, 35.9771385, -35.7303619, 36.0188293, -71.6302948, 71.7075043
16: -45.7617874, 33.7043076, -45.8439751, 33.8063622, -79.5681458, 79.5482788
17: -84.6841736, 23.2924137, -84.7306519, 23.3465328, -108.0307083, 108.0230637
18: -48.8449974, 31.1825829, -49.0537491, 31.3604565, -80.2054520, 80.2363281
19: -38.8020325, 18.4082985, -38.9419708, 18.5441055, -57.3461380, 57.3502693
20: -36.8172150, 23.2458763, -36.9028473, 23.3210773, -60.1382904, 60.1487236
21: -47.9832878, 21.9276218, -48.1323242, 22.0619812, -70.0452728, 70.0599442
22: -49.9216957, 22.0467510, -49.8782654, 22.0345402, -71.9562378, 71.9250183
23: -38.9678154, 23.8656273, -39.0474777, 23.9468822, -62.9146957, 62.9131050
24: -46.1976433, 23.9046402, -46.2648201, 23.9614334, -70.1590729, 70.1694641
25: -41.1185379, 24.6631012, -41.1495361, 24.6748638, -65.7934036, 65.8126373
26: -56.8467941, 33.3274117, -56.9118385, 33.4807968, -90.3275909, 90.2392502
27: -45.2342377, 28.6879463, -45.2846298, 28.7365608, -73.9707947, 73.9725800
28: -38.8620605, 26.6320133, -38.9105377, 26.6820450, -65.5441055, 65.5425491
29: -51.5536919, 20.5194912, -51.5167007, 20.5027657, -72.0564575, 72.0361938
30: -49.1582794, 26.0006981, -49.2429047, 26.0802574, -75.2385406, 75.2436066
31: -50.8691101, 27.6886845, -51.0549850, 27.8329315, -78.7020416, 78.7436676
32: -52.3126068, 24.6065350, -52.3315659, 24.6462936, -76.9589005, 76.9381027
33: -72.1468201, 33.6577759, -72.2277222, 33.7486267, -105.6185989, 105.5991058
34: -65.4106293, 17.0544319, -65.4592896, 17.1010990, -81.7056274, 81.6808472
35: -63.6471252, 23.4414711, -63.7206573, 23.5326080, -85.6485291, 85.5888290
36: -61.8467026, 24.3107624, -61.9046783, 24.3666821, -86.2133865, 86.2154388
37: -87.0508575, 19.7371235, -87.0218201, 19.7465801, -106.7974396, 106.7589417
38: -69.7716980, 29.0570145, -69.8391876, 29.1123695, -98.8840637, 98.8962021
39: -80.2668762, 30.5102463, -80.2928925, 30.5306053, -110.7974854, 110.8031387
40: -62.4703522, 25.5809498, -62.4504204, 25.5507050, -88.0210571, 88.0313721
41: -54.8867683, 32.7706757, -54.8750572, 32.7839622, -87.6707306, 87.6457367
42: -36.1974258, 25.8746605, -36.2176895, 25.9652367, -62.1626625, 62.0923500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
time: 86.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2363706
time: 81.26 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -56.3066597, 43.4200668, -56.6077728, 43.5357246, -99.8423843, 100.0278397
1: -25.0832291, 37.6451721, -25.3147221, 37.8279953, -62.9112244, 62.9598923
2: -21.5321541, 37.0470886, -21.8416176, 37.2641220, -58.7962761, 58.8887062
3: -24.0916290, 39.6084442, -24.4587631, 39.9002533, -63.9918823, 64.0672073
4: -28.1552887, 43.5887413, -28.5142918, 43.8294220, -71.9847107, 72.1030350
5: -24.3419571, 39.6094856, -24.6740799, 39.8356247, -64.1775818, 64.2835693
6: -54.1420670, 31.6967316, -54.2781448, 31.8636780, -86.0057449, 85.9748764
7: -30.1761131, 39.5175323, -30.4625187, 39.6350784, -69.8111877, 69.9800491
8: -36.3222771, 53.4007263, -36.6068420, 53.6664200, -89.9886932, 90.0075684
9: -28.9464817, 38.9178391, -29.1038303, 39.0431976, -67.9896774, 68.0216675
10: -49.2532883, 43.3129578, -49.6126900, 43.7746124, -93.0279007, 92.9256439
11: -48.6826210, 21.4579144, -49.1793556, 21.9146156, -70.5972366, 70.6372681
12: -55.0700226, 24.7050705, -55.3520927, 25.1711311, -79.1303024, 78.9211121
13: -50.2708473, 43.5184059, -50.5836487, 43.8085594, -94.0794067, 94.1020508
14: -87.1485214, 30.5721474, -87.5833206, 31.1180019, -118.2665253, 118.1554718
15: -35.4835892, 35.9071884, -35.7914276, 36.0499039, -71.5334930, 71.6986160
16: -45.6367645, 33.6052551, -45.9175110, 33.8975563, -79.5343170, 79.5227661
17: -84.5470276, 23.0445347, -84.9653778, 23.4429970, -107.9900208, 108.0099106
18: -48.6730003, 30.9845734, -49.0771904, 31.2934856, -79.9664841, 80.0617676
19: -38.6492691, 18.2246952, -38.9953804, 18.5134220, -57.1626892, 57.2200775
20: -36.6942406, 23.0800362, -36.9624252, 23.3472347, -60.0414734, 60.0424614
21: -47.8043747, 21.6995659, -48.2202415, 22.0624466, -69.8668213, 69.9198074
22: -49.8611794, 21.9111061, -50.0542336, 22.1128407, -71.9740219, 71.9653397
23: -38.8589783, 23.6803093, -39.1523094, 23.9933319, -62.8523102, 62.8326187
24: -46.1057663, 23.7735672, -46.3622437, 23.9976082, -70.1033783, 70.1358109
25: -41.0574799, 24.5172882, -41.2786713, 24.7720146, -65.8294983, 65.7959595
26: -56.6908684, 32.9968491, -57.0480652, 33.5059433, -90.1968079, 90.0449142
27: -45.0910187, 28.5892372, -45.3315201, 28.7391663, -73.8301849, 73.9207611
28: -38.7394714, 26.4434013, -38.9865913, 26.7175751, -65.4570465, 65.4299927
29: -51.4806900, 20.3407307, -51.7224083, 20.6010323, -72.0817261, 72.0631409
30: -49.0415802, 25.7727356, -49.3547516, 26.1549416, -75.1965179, 75.1274872
31: -50.6944199, 27.4904537, -51.1198807, 27.8324280, -78.5268478, 78.6103363
32: -52.2317085, 24.5117283, -52.3873177, 24.6617908, -76.8935013, 76.8990479
33: -71.9656601, 33.5419235, -72.2894287, 33.8211136, -105.4837799, 105.6633377
34: -65.3379135, 16.9668198, -65.4961548, 17.1383648, -81.5797348, 81.7014084
35: -63.5882492, 23.3802185, -63.7659073, 23.5553551, -85.5710678, 85.6488495
36: -61.7927704, 24.2506180, -61.9614563, 24.3927097, -86.1854782, 86.2120743
37: -86.9660492, 19.6487904, -87.1452942, 19.8216400, -106.7876892, 106.7940826
38: -69.6726227, 29.0188503, -69.9179535, 29.2138634, -98.8864899, 98.9368057
39: -80.1315308, 30.4518356, -80.4005432, 30.6483173, -110.7798462, 110.8523788
40: -62.3826828, 25.5629501, -62.5575981, 25.7123985, -88.0950775, 88.1205444
41: -54.8138733, 32.7081642, -54.9758720, 32.8700485, -87.6839218, 87.6840363
42: -36.1310120, 25.7727699, -36.2253532, 25.9923134, -62.1233253, 61.9981232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2302435
time: 89.85 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2340991
time: 90.11 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -56.4954185, 43.4980698, -56.6630974, 43.5496368, -100.0450592, 100.1611633
1: -25.2240105, 37.7370758, -25.3577232, 37.8369598, -63.0609703, 63.0947990
2: -21.7428226, 37.1661072, -21.9101009, 37.2742081, -59.0170288, 59.0762100
3: -24.3322411, 39.7481270, -24.5372849, 39.9124680, -64.2447052, 64.2854156
4: -28.3855152, 43.7099380, -28.5889320, 43.8398056, -72.2253189, 72.2988739
5: -24.5644417, 39.7319031, -24.7453861, 39.8478012, -64.4122467, 64.4772873
6: -54.2238159, 31.7871723, -54.2965508, 31.8900661, -86.1138840, 86.0837250
7: -30.3598747, 39.5832214, -30.5178528, 39.6438522, -70.0037231, 70.1010742
8: -36.5072708, 53.5332336, -36.6654472, 53.6808357, -90.1881104, 90.1986847
9: -29.0426788, 38.9897003, -29.1334305, 39.0611420, -68.1038208, 68.1231308
10: -49.3834763, 43.5298843, -49.6383400, 43.8401108, -93.2235870, 93.1682281
11: -48.9129639, 21.7421875, -49.2006760, 22.0105724, -70.9235382, 70.9428635
12: -55.2194977, 25.0119629, -55.3667488, 25.2708359, -79.3826752, 79.2311859
13: -50.4111443, 43.6577682, -50.6288376, 43.8351364, -94.2462769, 94.2866058
14: -87.3519745, 30.9006233, -87.6136169, 31.2287178, -118.5806885, 118.5142365
15: -35.6294708, 35.9842644, -35.8337708, 36.0671577, -71.6966248, 71.8180389
16: -45.7741241, 33.7338028, -45.9519958, 33.9350204, -79.7091446, 79.6857986
17: -84.7546616, 23.2992592, -84.9846725, 23.5278454, -108.2825089, 108.2839355
18: -48.8606415, 31.1777210, -49.0935783, 31.3589821, -80.2196198, 80.2713013
19: -38.8156662, 18.4101105, -39.0153885, 18.5753269, -57.3909912, 57.4254990
20: -36.8343201, 23.2505131, -36.9840164, 23.4039574, -60.2382774, 60.2345276
21: -48.0082245, 21.9318752, -48.2429733, 22.1405754, -70.1488037, 70.1748505
22: -49.9746170, 22.0528202, -50.0728073, 22.1579781, -72.1325989, 72.1256256
23: -38.9994507, 23.8698597, -39.1696053, 24.0557728, -63.0552216, 63.0394669
24: -46.2242470, 23.9084587, -46.3806572, 24.0422707, -70.2665176, 70.2891159
25: -41.1582184, 24.6737480, -41.2957802, 24.8237171, -65.9819336, 65.9695282
26: -56.8848038, 33.3310509, -57.0701485, 33.6168861, -90.5016937, 90.4011993
27: -45.2439384, 28.6922245, -45.3543129, 28.7749557, -74.0188904, 74.0465393
28: -38.8838844, 26.6365166, -39.0067825, 26.7815056, -65.6653900, 65.6432953
29: -51.6157036, 20.5252953, -51.7391663, 20.6621647, -72.2778702, 72.2644653
30: -49.1963196, 26.0087547, -49.3731499, 26.2324352, -75.4287567, 75.3819046
31: -50.8881073, 27.6926193, -51.1438065, 27.9001408, -78.7882462, 78.8364258
32: -52.3266182, 24.6114521, -52.4078560, 24.6910973, -77.0177155, 77.0193100
33: -72.1573639, 33.6817627, -72.3509674, 33.8430367, -105.6993103, 105.8748016
34: -65.4188538, 17.0634155, -65.5181198, 17.1626816, -81.7086334, 81.8282547
35: -63.6554604, 23.4486103, -63.7849541, 23.5693722, -85.6669159, 85.7535248
36: -61.8555298, 24.3163719, -61.9761124, 24.4089813, -86.2645111, 86.2924805
37: -87.0653076, 19.7595558, -87.1663361, 19.8560543, -106.9213638, 106.9258881
38: -69.7820511, 29.0879498, -69.9467545, 29.2284851, -99.0105362, 99.0347061
39: -80.2786255, 30.5498810, -80.4461060, 30.6632156, -110.9418411, 110.9959869
40: -62.4800377, 25.6332741, -62.5801239, 25.7269402, -88.2069778, 88.2133942
41: -54.8949509, 32.8004036, -54.9970856, 32.8952484, -87.7901993, 87.7974854
42: -36.1981506, 25.8868999, -36.2408600, 26.0242176, -62.2223663, 62.1277618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
time: 94.16 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2363706
time: 106.99 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -56.5033722, 43.4555054, -56.5128593, 43.4542694, -99.9576416, 99.9683685
1: -25.2490540, 37.7253418, -25.2626038, 37.7416611, -62.9907150, 62.9879456
2: -21.7430973, 37.1449547, -21.7909107, 37.2057877, -58.9488831, 58.9358673
3: -24.3589439, 39.7735596, -24.4922924, 39.8608170, -64.2197571, 64.2658539
4: -28.4104996, 43.6883163, -28.4209747, 43.6993637, -72.1098633, 72.1092911
5: -24.5723763, 39.7108307, -24.6455059, 39.7433891, -64.3157654, 64.3563385
6: -54.2250671, 31.7524414, -54.2366791, 31.7960205, -86.0210876, 85.9891205
7: -30.3671417, 39.5484848, -30.3746872, 39.5120087, -69.8791504, 69.9231720
8: -36.5218773, 53.5329933, -36.5593948, 53.5758133, -90.0976868, 90.0923920
9: -29.0476074, 38.9996300, -29.0772018, 39.0084267, -68.0560303, 68.0768280
10: -49.4996071, 43.7156525, -49.5132370, 43.7337875, -93.2333984, 93.2288895
11: -48.9511108, 21.8202171, -49.0623856, 21.9088402, -70.8599548, 70.8825989
12: -55.1863289, 25.0418892, -55.2034302, 25.1061840, -79.1792450, 79.1099930
13: -50.4931030, 43.7103577, -50.5869293, 43.7694054, -94.2625122, 94.2972870
14: -87.3646393, 31.0159836, -87.3437042, 30.9983902, -118.3630295, 118.3596878
15: -35.6626854, 35.9993286, -35.7485123, 36.0253143, -71.6880035, 71.7478409
16: -45.8327026, 33.7628250, -45.8564453, 33.8257523, -79.6584549, 79.6192703
17: -84.7244339, 23.3406582, -84.7351990, 23.3602962, -108.0847321, 108.0758591
18: -48.9039001, 31.2641773, -49.0605850, 31.3903255, -80.2942276, 80.3247604
19: -38.8542595, 18.4432373, -38.9495583, 18.5551777, -57.4094391, 57.3927956
20: -36.8470917, 23.2800865, -36.9079590, 23.3312645, -60.1783562, 60.1880455
21: -48.0343132, 21.9752216, -48.1385002, 22.0762577, -70.1105728, 70.1137238
22: -49.9235077, 22.0372868, -49.8839836, 22.0298138, -71.9533234, 71.9212723
23: -39.0153427, 23.9233208, -39.0536194, 23.9637909, -62.9791336, 62.9769402
24: -46.2528839, 23.9485321, -46.2722549, 23.9759884, -70.2288742, 70.2207870
25: -41.1725731, 24.7052689, -41.1574326, 24.6875858, -65.8601608, 65.8627014
26: -56.8551521, 33.3698883, -56.9177971, 33.4912567, -90.3464050, 90.2876892
27: -45.2105675, 28.6680470, -45.2911949, 28.7252235, -73.9357910, 73.9592438
28: -38.8628845, 26.6270790, -38.9159203, 26.6782608, -65.5411453, 65.5429993
29: -51.5581551, 20.5137997, -51.5208321, 20.4990234, -72.0571747, 72.0346298
30: -49.1997986, 26.0598717, -49.2493858, 26.1018677, -75.3016663, 75.3092575
31: -50.9566879, 27.7672634, -51.0652466, 27.8594532, -78.8161392, 78.8325119
32: -52.3157654, 24.5886803, -52.3365479, 24.6395035, -76.9552689, 76.9252319
33: -72.2117767, 33.6959000, -72.2467957, 33.7486191, -105.6834564, 105.6252670
34: -65.4528656, 17.0716724, -65.4727936, 17.1034336, -81.7553864, 81.6932220
35: -63.7264595, 23.5063763, -63.7460365, 23.5366077, -85.7442551, 85.6782455
36: -61.9324532, 24.3529720, -61.9322166, 24.3671875, -86.2996368, 86.2851868
37: -87.0729523, 19.7268295, -87.0327301, 19.7409477, -106.8139038, 106.7595596
38: -69.8673859, 29.1391582, -69.8699951, 29.1196136, -98.9869995, 99.0091553
39: -80.3341293, 30.5426559, -80.3118134, 30.5326023, -110.8667297, 110.8544693
40: -62.4958649, 25.5844307, -62.4602051, 25.5477180, -88.0435791, 88.0446320
41: -54.9346466, 32.7696457, -54.8874741, 32.7816772, -87.7163239, 87.6571198
42: -36.1997681, 25.8863430, -36.2204971, 25.9658337, -62.1656036, 62.1068420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
time: 95.86 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2352701
time: 90.16 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -56.6941757, 43.5346565, -56.5685883, 43.4683762, -100.1625519, 100.1032410
1: -25.3918457, 37.8186722, -25.3060074, 37.7508621, -63.1427078, 63.1246796
2: -21.9559402, 37.2655869, -21.8598709, 37.2160912, -59.1720314, 59.1254578
3: -24.6019211, 39.9156494, -24.5711880, 39.8733330, -64.4752502, 64.4868393
4: -28.6439323, 43.8117256, -28.4961281, 43.7100601, -72.3539886, 72.3078537
5: -24.7967033, 39.8347969, -24.7172585, 39.7557945, -64.5524979, 64.5520554
6: -54.3097458, 31.8480949, -54.2552567, 31.8227501, -86.1324921, 86.1033478
7: -30.5539513, 39.6146088, -30.4305038, 39.5209045, -70.0748596, 70.0451126
8: -36.7097778, 53.6671753, -36.6185455, 53.5904884, -90.3002625, 90.2857208
9: -29.1483402, 39.0744133, -29.1073875, 39.0268211, -68.1751633, 68.1818008
10: -49.6347046, 43.9363861, -49.5394897, 43.7999992, -93.4347076, 93.4758759
11: -49.1866493, 22.1076660, -49.0841751, 22.0050278, -71.1916809, 71.1918411
12: -55.3361816, 25.3513603, -55.2181740, 25.2065773, -79.4323730, 79.4235840
13: -50.6413841, 43.8496628, -50.6325378, 43.7961388, -94.4375229, 94.4822006
14: -87.5709000, 31.3466206, -87.3743362, 31.1093693, -118.6802673, 118.7209549
15: -35.8282928, 36.0787544, -35.7960091, 36.0428886, -71.8711853, 71.8747635
16: -45.9774132, 33.8929138, -45.8919144, 33.8623390, -79.8397522, 79.7848282
17: -84.9333496, 23.5995026, -84.7547531, 23.4455528, -108.3789062, 108.3542557
18: -49.0940590, 31.4715805, -49.0773239, 31.4565468, -80.5506058, 80.5489044
19: -39.0243759, 18.6303787, -38.9700241, 18.6173420, -57.6417160, 57.6004028
20: -36.9888954, 23.4516830, -36.9298248, 23.3881550, -60.3770523, 60.3815079
21: -48.2414360, 22.2089214, -48.1616249, 22.1546097, -70.3960419, 70.3705444
22: -50.0372162, 22.1833687, -49.9029007, 22.0754757, -72.1126938, 72.0862732
23: -39.1579933, 24.1149750, -39.0712967, 24.0265083, -63.1845016, 63.1862717
24: -46.3730125, 24.0843391, -46.2910461, 24.0206718, -70.3936844, 70.3753815
25: -41.2756844, 24.8638744, -41.1749649, 24.7394447, -66.0151291, 66.0388412
26: -57.0513153, 33.7094803, -56.9402351, 33.6030426, -90.6543579, 90.6497192
27: -45.3671646, 28.7837181, -45.3142281, 28.7676735, -74.1348419, 74.0979462
28: -39.0086708, 26.8223171, -38.9364052, 26.7424927, -65.7511597, 65.7587204
29: -51.6941681, 20.7004375, -51.5377502, 20.5603905, -72.2545624, 72.2381897
30: -49.3582916, 26.2987289, -49.2682114, 26.1794987, -75.5377884, 75.5669403
31: -51.1539497, 27.9710560, -51.0896263, 27.9273586, -79.0813065, 79.0606842
32: -52.4114342, 24.6915359, -52.3576317, 24.6693058, -77.0807419, 77.0491638
33: -72.4055634, 33.8383598, -72.3085098, 33.7707405, -105.9020538, 105.8398209
34: -65.5375214, 17.1735554, -65.4952469, 17.1286678, -81.8827667, 81.8294830
35: -63.7957153, 23.5770893, -63.7652702, 23.5508842, -85.8503342, 85.7837830
36: -61.9979897, 24.4201107, -61.9468918, 24.3837032, -86.3816910, 86.3670044
37: -87.1762314, 19.8450336, -87.0537338, 19.7765484, -106.9527817, 106.8987656
38: -69.9817200, 29.2116222, -69.8991241, 29.1351528, -99.1168747, 99.1107483
39: -80.4876328, 30.6424599, -80.3580704, 30.5476418, -111.0352783, 111.0005341
40: -62.5962982, 25.6568356, -62.4825897, 25.5615139, -88.1578140, 88.1394272
41: -55.0190811, 32.8619003, -54.9088516, 32.8059540, -87.8250351, 87.7707520
42: -36.2694702, 26.0081902, -36.2362213, 25.9983902, -62.2678604, 62.2444115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2366513
time: 122.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2379575
time: 96.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -56.5162506, 43.4806213, -56.6655426, 43.5511932, -100.0674438, 100.1461639
1: -25.2562790, 37.7544975, -25.3667240, 37.8405457, -63.0968246, 63.1212234
2: -21.7513008, 37.1648598, -21.9116840, 37.2775993, -59.0289001, 59.0765457
3: -24.3627491, 39.7854958, -24.5480824, 39.9192390, -64.2819901, 64.3335800
4: -28.4219704, 43.7307587, -28.5988655, 43.8430099, -72.2649841, 72.3296204
5: -24.5787392, 39.7393188, -24.7493629, 39.8533401, -64.4320831, 64.4886780
6: -54.2315216, 31.7711010, -54.3034363, 31.8844528, -86.1159744, 86.0745392
7: -30.3774185, 39.5867386, -30.5220699, 39.6473694, -70.0247879, 70.1088104
8: -36.5281715, 53.5635147, -36.6718216, 53.6874619, -90.2156372, 90.2353363
9: -29.0540695, 39.0091743, -29.1346817, 39.0676117, -68.1216812, 68.1438599
10: -49.5340958, 43.7245102, -49.6477966, 43.9080353, -93.4421310, 93.3723068
11: -48.9897728, 21.8266678, -49.2076340, 22.0409985, -71.0307693, 71.0343018
12: -55.2332535, 25.0480728, -55.3717728, 25.2829590, -79.4087601, 79.2702484
13: -50.5008621, 43.7214966, -50.6580887, 43.8404922, -94.3413544, 94.3795853
14: -87.4401703, 31.0208111, -87.6232224, 31.2708073, -118.7109756, 118.6440353
15: -35.6764565, 36.0063248, -35.8502655, 36.0740318, -71.7504883, 71.8565903
16: -45.8448563, 33.7881775, -45.9652786, 33.9524231, -79.7972794, 79.7534561
17: -84.7955170, 23.3473148, -84.9889221, 23.5433846, -108.3388977, 108.3362350
18: -48.9207230, 31.2506866, -49.1006012, 31.3853817, -80.3061066, 80.3512878
19: -38.8678360, 18.4448147, -39.0233345, 18.5869522, -57.4547882, 57.4681473
20: -36.8634033, 23.2848015, -36.9893951, 23.4149761, -60.2783813, 60.2741966
21: -48.0591049, 21.9794655, -48.2495804, 22.1558552, -70.2149582, 70.2290497
22: -49.9759598, 22.0432892, -50.0777855, 22.1547108, -72.1306686, 72.1210785
23: -39.0470505, 23.9274998, -39.1760788, 24.0741463, -63.1211967, 63.1035767
24: -46.2797089, 23.9523983, -46.3880844, 24.0577431, -70.3374481, 70.3404846
25: -41.2120628, 24.7161255, -41.3033524, 24.8376465, -66.0497131, 66.0194778
26: -56.8933716, 33.3731689, -57.0755081, 33.6289368, -90.5223083, 90.4486771
27: -45.2203293, 28.6735306, -45.3616257, 28.7675896, -73.9879150, 74.0351562
28: -38.8840714, 26.6315327, -39.0125198, 26.7784157, -65.6624908, 65.6440506
29: -51.6195221, 20.5197678, -51.7426376, 20.6594162, -72.2789383, 72.2624054
30: -49.2349014, 26.0679665, -49.3795662, 26.2506561, -75.4855576, 75.4475327
31: -50.9756508, 27.7711067, -51.1546402, 27.9275417, -78.9031906, 78.9257507
32: -52.3276405, 24.5937920, -52.4113731, 24.6844635, -77.0121002, 77.0051651
33: -72.2225494, 33.7185593, -72.3710480, 33.8431549, -105.7468185, 105.8969193
34: -65.4601440, 17.0803242, -65.5319366, 17.1657715, -81.7453308, 81.8562622
35: -63.7341080, 23.5141125, -63.8114014, 23.5735950, -85.7589722, 85.8552399
36: -61.9407616, 24.3605595, -62.0075264, 24.4097748, -86.3505402, 86.3680878
37: -87.0875626, 19.7572536, -87.1779022, 19.8502827, -106.9378433, 106.9351578
38: -69.8770981, 29.1722527, -69.9793625, 29.2363796, -99.1134796, 99.1516113
39: -80.3454590, 30.5817299, -80.4677811, 30.6652336, -111.0106964, 111.0495148
40: -62.5057030, 25.6328621, -62.5911560, 25.7207489, -88.2264557, 88.2240143
41: -54.9428215, 32.7975006, -55.0134506, 32.8909302, -87.8337555, 87.8109512
42: -36.2009430, 25.8989067, -36.2440605, 26.0259094, -62.2268524, 62.1429672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
time: 83.33 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1342810, upper bound: 44.2352701
time: 148.32 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -56.7070808, 43.5596695, -56.7211151, 43.5650978, -100.2721786, 100.2807846
1: -25.3989944, 37.8478470, -25.4099274, 37.8496132, -63.2486076, 63.2577744
2: -21.9640446, 37.2854729, -21.9804287, 37.2877426, -59.2517853, 59.2658997
3: -24.6057472, 39.9275970, -24.6267719, 39.9316025, -64.5373535, 64.5543671
4: -28.6553974, 43.8541794, -28.6737938, 43.8534775, -72.5088730, 72.5279694
5: -24.8029728, 39.8632736, -24.8209000, 39.8655930, -64.6685638, 64.6841736
6: -54.3162880, 31.8674431, -54.3218689, 31.9108734, -86.2271576, 86.1893158
7: -30.5641422, 39.6528778, -30.5776596, 39.6561890, -70.2203293, 70.2305374
8: -36.7160378, 53.6977959, -36.7306671, 53.7019653, -90.4179993, 90.4284668
9: -29.1550159, 39.0838890, -29.1644993, 39.0856857, -68.2406998, 68.2483902
10: -49.6692009, 43.9450912, -49.6736908, 43.9738388, -93.6430359, 93.6187820
11: -49.2253838, 22.1141396, -49.2291031, 22.1371880, -71.3625717, 71.3432465
12: -55.3830795, 25.3574181, -55.3864365, 25.3829269, -79.6614761, 79.5847626
13: -50.6487808, 43.8609657, -50.7034645, 43.8669815, -94.5157623, 94.5644302
14: -87.6463928, 31.3515682, -87.6534882, 31.3816757, -119.0280685, 119.0050583
15: -35.8431015, 36.0857849, -35.8958893, 36.0914040, -71.9345093, 71.9816742
16: -45.9895554, 33.9191360, -46.0000534, 33.9884491, -79.9780045, 79.9191895
17: -85.0044403, 23.6062241, -85.0082932, 23.6284561, -108.6328964, 108.6145172
18: -49.1110268, 31.4580212, -49.1171150, 31.4512863, -80.5623169, 80.5751343
19: -39.0379562, 18.6318474, -39.0434952, 18.6489735, -57.6869278, 57.6753426
20: -37.0052261, 23.4563332, -37.0110321, 23.4718323, -60.4770584, 60.4673653
21: -48.2662392, 22.2130699, -48.2724037, 22.2341137, -70.5003510, 70.4854736
22: -50.0897255, 22.1893559, -50.0963326, 22.2000961, -72.2898254, 72.2856903
23: -39.1897049, 24.1191101, -39.1934814, 24.1367702, -63.3264771, 63.3125916
24: -46.3997726, 24.0882607, -46.4065399, 24.1024799, -70.5022507, 70.4947968
25: -41.3151360, 24.8747654, -41.3206558, 24.8894596, -66.2045975, 66.1954193
26: -57.0894737, 33.7127228, -57.0976639, 33.7403069, -90.8297806, 90.8103867
27: -45.3770866, 28.7856503, -45.3843956, 28.8035622, -74.1806488, 74.1700439
28: -39.0298538, 26.8267136, -39.0328217, 26.8425369, -65.8723907, 65.8595352
29: -51.7555199, 20.7063923, -51.7593956, 20.7206917, -72.4762115, 72.4657898
30: -49.3934441, 26.3067989, -49.3981247, 26.3283787, -75.7218246, 75.7049255
31: -51.1729507, 27.9748611, -51.1787186, 27.9953709, -79.1683197, 79.1535797
32: -52.4231644, 24.6964302, -52.4318581, 24.7138710, -77.1370392, 77.1282883
33: -72.4163513, 33.8609772, -72.4326859, 33.8650970, -105.9650497, 106.1113434
34: -65.5451584, 17.1821671, -65.5536880, 17.1903496, -81.8793182, 81.9869232
35: -63.8035355, 23.5848141, -63.8305244, 23.5876789, -85.8642273, 85.9634476
36: -62.0065155, 24.4276466, -62.0226440, 24.4260979, -86.4326172, 86.4502869
37: -87.1910095, 19.8696346, -87.1990814, 19.8848553, -107.0758667, 107.0687180
38: -69.9915924, 29.2437229, -70.0086823, 29.2511292, -99.2427216, 99.2524033
39: -80.4989166, 30.6815834, -80.5139465, 30.6800137, -111.1789322, 111.1955261
40: -62.6064491, 25.7057056, -62.6136894, 25.7354050, -88.3418579, 88.3193970
41: -55.0273514, 32.8927002, -55.0348129, 32.9161377, -87.9434891, 87.9275131
42: -36.2707100, 26.0212631, -36.2596359, 26.0584888, -62.3292007, 62.2808990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2366513
time: 129.56 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2379575
time: 95.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 227.34 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2301780
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2338611
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2363706
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2302435
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2340991
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2363706
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2352701
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2366513
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2379575
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -44.1342810, upper bound: 44.2352701
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2366513
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 227.34
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2379575

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -56.2755737, 43.3435745, -56.3226280, 43.2826805, -99.5582581, 99.6661987
1: -25.0655937, 37.5650826, -25.1135349, 37.5795288, -62.6451225, 62.6786194
2: -21.5141792, 36.9878464, -21.6180706, 37.0800362, -58.5942154, 58.6059189
3: -24.0811901, 39.5710754, -24.3399906, 39.7607040, -63.8418961, 63.9110641
4: -28.1322002, 43.4762268, -28.1937332, 43.4863625, -71.6185608, 71.6699600
5: -24.3264027, 39.5394211, -24.4818420, 39.5980034, -63.9244080, 64.0212631
6: -54.1260300, 31.6484680, -54.1695633, 31.6729889, -85.7990189, 85.8180313
7: -30.1518440, 39.4170456, -30.1903725, 39.3180313, -69.4698792, 69.6074219
8: -36.3066864, 53.3115692, -36.3849640, 53.3823318, -89.6890182, 89.6965332
9: -28.9324188, 38.8826065, -28.9970589, 38.8957596, -67.8281784, 67.8796692
10: -49.1943550, 43.2864151, -49.3908081, 43.5100403, -92.7043915, 92.6772232
11: -48.5867004, 21.4418640, -48.8623352, 21.6884766, -70.2751770, 70.3041992
12: -54.9590683, 24.6877537, -54.9914284, 24.8543568, -78.6908188, 78.5555649
13: -50.2366333, 43.4944687, -50.4127655, 43.6854973, -93.9221344, 93.9072342
14: -87.0078812, 30.5590763, -87.0866699, 30.7086830, -117.7165680, 117.6457443
15: -35.4388008, 35.8899384, -35.5727654, 35.9632912, -71.4020920, 71.4627075
16: -45.6062164, 33.5212479, -45.7288780, 33.5846519, -79.1908722, 79.2501221
17: -84.3910828, 23.0264263, -84.4471741, 23.1410351, -107.5321198, 107.4736023
18: -48.6407089, 30.9628487, -49.0214577, 31.2297459, -79.8704529, 79.9843063
19: -38.6037674, 18.2183495, -38.8142014, 18.4299126, -57.0336800, 57.0325508
20: -36.6490326, 23.0697765, -36.7800980, 23.1993828, -59.8484154, 59.8498764
21: -47.7290878, 21.6888809, -47.9456444, 21.8975182, -69.6266022, 69.6345215
22: -49.7217255, 21.8976135, -49.5978813, 21.8858109, -71.6075363, 71.4954987
23: -38.7820282, 23.6700211, -38.8864822, 23.7975082, -62.5795364, 62.5565033
24: -46.0407829, 23.7647324, -46.1220474, 23.8615189, -69.9022980, 69.8867798
25: -40.9625626, 24.4969406, -40.9629822, 24.5181866, -65.4807510, 65.4599228
26: -56.5947685, 32.9851151, -56.7005005, 33.2473297, -89.8421021, 89.6856155
27: -45.0654488, 28.5655384, -45.2047958, 28.6459618, -73.7114105, 73.7703323
28: -38.6816559, 26.4331188, -38.7689209, 26.5342751, -65.2159271, 65.2020416
29: -51.3178139, 20.3286381, -51.1944008, 20.3163471, -71.6341629, 71.5230408
30: -48.9477425, 25.7542133, -49.0538254, 25.8925362, -74.8402786, 74.8080368
31: -50.6440163, 27.4800739, -50.9245453, 27.7031860, -78.3471985, 78.4046173
32: -52.1941376, 24.4990673, -52.2254639, 24.5747700, -76.7689056, 76.7245331
33: -71.9383698, 33.5049667, -72.0824814, 33.6712723, -105.3408051, 105.2319183
34: -65.3127441, 16.9454727, -65.3766785, 17.0161610, -81.5035934, 81.4455185
35: -63.5527573, 23.3625488, -63.6092873, 23.4804459, -85.4986572, 85.3755188
36: -61.7607460, 24.2361813, -61.8086090, 24.3058910, -86.0666351, 86.0447922
37: -86.9322662, 19.5996628, -86.9476166, 19.6281128, -106.5603790, 106.5472794
38: -69.6454163, 28.9556293, -69.7496796, 29.0016632, -98.6470795, 98.7053070
39: -80.1013489, 30.3747482, -80.1509552, 30.3991127, -110.5004578, 110.5257034
40: -62.3584251, 25.4463043, -62.3663216, 25.3546410, -87.7130661, 87.8126221
41: -54.7950287, 32.6516991, -54.7964783, 32.6609039, -87.4559326, 87.4481812
42: -36.1163635, 25.7466259, -36.1457443, 25.8790073, -61.9953690, 61.8923721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1868988
time: 111.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2301780
time: 83.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -56.2926140, 43.3934517, -56.4507599, 43.4324799, -99.7250977, 99.8442078
1: -25.0752392, 37.6148987, -25.2083492, 37.7243462, -62.7995834, 62.8232498
2: -21.5232430, 37.0250778, -21.7184258, 37.1888199, -58.7120628, 58.7435036
3: -24.0873566, 39.5960846, -24.4023933, 39.8388062, -63.9261627, 63.9984779
4: -28.1431885, 43.5409737, -28.3320637, 43.6804276, -71.8236160, 71.8730392
5: -24.3349571, 39.5802689, -24.5683403, 39.7220306, -64.0569916, 64.1486053
6: -54.1351242, 31.6751537, -54.2099915, 31.7618256, -85.8969498, 85.8851471
7: -30.1648006, 39.4775238, -30.3119221, 39.4925613, -69.6573639, 69.7894440
8: -36.3152275, 53.3687134, -36.4923935, 53.5485535, -89.8637848, 89.8611069
9: -28.9395142, 38.9059296, -29.0458794, 38.9734039, -67.9129181, 67.9518127
10: -49.2166214, 43.3023911, -49.4707336, 43.5979042, -92.8145294, 92.7731247
11: -48.6406097, 21.4509010, -49.0287704, 21.7821026, -70.4227142, 70.4796753
12: -55.0216827, 24.6980324, -55.1774445, 24.9924088, -78.8974304, 78.7407761
13: -50.2590179, 43.5073357, -50.4972229, 43.7342377, -93.9932556, 94.0045624
14: -87.0712585, 30.5665035, -87.2967834, 30.8455429, -117.9168015, 117.8632889
15: -35.4623642, 35.8993454, -35.6727066, 35.9991264, -71.4614868, 71.5720520
16: -45.6234512, 33.5695114, -45.8059311, 33.7374344, -79.3608856, 79.3754425
17: -84.4748459, 23.0366745, -84.7038574, 23.2578831, -107.7327271, 107.7405319
18: -48.6565857, 30.9767838, -49.0331039, 31.2641754, -79.9207611, 80.0098877
19: -38.6347504, 18.2225037, -38.9183464, 18.4808693, -57.1156197, 57.1408501
20: -36.6763916, 23.0750561, -36.8780060, 23.2629566, -59.9393463, 59.9530640
21: -47.7782173, 21.6949272, -48.1044655, 21.9824677, -69.7606812, 69.7993927
22: -49.8058586, 21.9043159, -49.8497620, 21.9867020, -71.7925568, 71.7540741
23: -38.8263664, 23.6756020, -39.0262260, 23.8825970, -62.7089615, 62.7018280
24: -46.0777550, 23.7695618, -46.2408371, 23.9158459, -69.9935989, 70.0103989
25: -41.0165215, 24.5058308, -41.1267433, 24.6200657, -65.6365891, 65.6325760
26: -56.6514778, 32.9923935, -56.8831444, 33.3667603, -90.0182343, 89.8755341
27: -45.0804443, 28.5791473, -45.2577133, 28.6837673, -73.7642136, 73.8368607
28: -38.7168884, 26.4383850, -38.8870087, 26.6160870, -65.3329773, 65.3253937
29: -51.4161491, 20.3343945, -51.4892540, 20.4394646, -71.8556137, 71.8236465
30: -49.0021515, 25.7639141, -49.2188225, 25.9997387, -75.0018921, 74.9827347
31: -50.6745529, 27.4861832, -51.0272102, 27.7637749, -78.4383240, 78.5133972
32: -52.2165031, 24.5063057, -52.3020630, 24.6150150, -76.8315201, 76.8083649
33: -71.9526215, 33.5174446, -72.1550369, 33.7246552, -105.4115448, 105.4336243
34: -65.3268356, 16.9573174, -65.4301758, 17.0744801, -81.5680008, 81.5661469
35: -63.5765190, 23.3726807, -63.6862144, 23.5171909, -85.5412979, 85.4613419
36: -61.7823257, 24.2446175, -61.8822937, 24.3487492, -86.1310730, 86.1269073
37: -86.9505768, 19.6187115, -86.9970703, 19.6897888, -106.6403656, 106.6157837
38: -69.6611938, 28.9845657, -69.8052292, 29.0832653, -98.7444611, 98.7897949
39: -80.1184616, 30.4113865, -80.2424545, 30.5121021, -110.6305618, 110.6538391
40: -62.3719139, 25.5072231, -62.4239845, 25.5136337, -87.8855438, 87.9312057
41: -54.8051109, 32.6778946, -54.8516846, 32.7436218, -87.5487366, 87.5295792
42: -36.1275902, 25.7598381, -36.1917114, 25.9300232, -62.0576134, 61.9515495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1204245
time: 80.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2338611
time: 103.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -56.4643173, 43.4211006, -56.3778954, 43.2969666, -99.7612839, 99.7989960
1: -25.2065315, 37.6569786, -25.1568127, 37.5886917, -62.7952232, 62.8137894
2: -21.7249012, 37.1069679, -21.6870155, 37.0903702, -58.8152695, 58.7939835
3: -24.3219032, 39.7107697, -24.4190750, 39.7731590, -64.0950623, 64.1298447
4: -28.3626270, 43.5960426, -28.2689552, 43.4968910, -71.8595200, 71.8649979
5: -24.5490055, 39.6618347, -24.5537510, 39.6103630, -64.1593704, 64.2155838
6: -54.2076073, 31.7386780, -54.1880798, 31.6999664, -85.9075775, 85.9267578
7: -30.3359108, 39.4829063, -30.2461281, 39.3271103, -69.6630249, 69.7290344
8: -36.4917717, 53.4441566, -36.4441681, 53.3970032, -89.8887787, 89.8883209
9: -29.0285358, 38.9545288, -29.0260353, 38.9147339, -67.9432678, 67.9805603
10: -49.3245316, 43.5035172, -49.4167442, 43.5765076, -92.9010391, 92.9202576
11: -48.8171463, 21.7262344, -48.8842010, 21.7845268, -70.6016693, 70.6104355
12: -55.1083336, 24.9949284, -55.0062027, 24.9551201, -78.9427643, 78.8568420
13: -50.3804131, 43.6334915, -50.4625931, 43.7121010, -94.0925140, 94.0960846
14: -87.2095032, 30.8874664, -87.1170731, 30.8200397, -118.0295410, 118.0045395
15: -35.5874252, 35.9671440, -35.6209412, 35.9807053, -71.5681305, 71.5880890
16: -45.7437096, 33.6467247, -45.7637329, 33.6217194, -79.3654327, 79.4104614
17: -84.5985794, 23.2813721, -84.4668274, 23.2261734, -107.8247528, 107.7481995
18: -48.8279953, 31.1738167, -49.0383148, 31.3066769, -80.1346741, 80.2121277
19: -38.7703819, 18.4036846, -38.8346481, 18.4921188, -57.2625008, 57.2383347
20: -36.7891388, 23.2403221, -36.8019524, 23.2563019, -60.0454407, 60.0422745
21: -47.9330406, 21.9209251, -47.9686852, 21.9758091, -69.9088516, 69.8896103
22: -49.8334389, 22.0395012, -49.6173592, 21.9311943, -71.7646332, 71.6568604
23: -38.9223289, 23.8594894, -38.9040794, 23.8601456, -62.7824745, 62.7635689
24: -46.1596375, 23.8995018, -46.1411819, 23.9059658, -70.0656052, 70.0406799
25: -41.0637665, 24.6534729, -40.9807663, 24.5697327, -65.6334991, 65.6342392
26: -56.7889328, 33.3193283, -56.7229996, 33.3586998, -90.1476288, 90.0423279
27: -45.2180138, 28.6793556, -45.2281227, 28.6939850, -73.9120026, 73.9074783
28: -38.8261337, 26.6263485, -38.7894363, 26.5984230, -65.4245605, 65.4157867
29: -51.4523544, 20.5132332, -51.2112236, 20.3777218, -71.8300781, 71.7244568
30: -49.1026344, 25.9904022, -49.0727005, 25.9700127, -75.0726471, 75.0631027
31: -50.8378601, 27.6821022, -50.9489288, 27.7706680, -78.6085281, 78.6310272
32: -52.2880325, 24.5991058, -52.2466354, 24.6048126, -76.8928452, 76.8457413
33: -72.1303864, 33.6448288, -72.1442795, 33.6934395, -105.5375366, 105.4449921
34: -65.3949814, 17.0421371, -65.3997650, 17.0404968, -81.6169128, 81.5785828
35: -63.6197472, 23.4308319, -63.6285400, 23.4946671, -85.5995102, 85.4831696
36: -61.8232574, 24.3018646, -61.8236923, 24.3223133, -86.1455688, 86.1255569
37: -87.0312653, 19.7146702, -86.9681244, 19.6651058, -106.6963730, 106.6827927
38: -69.7548065, 29.0247021, -69.7787476, 29.0169888, -98.7717972, 98.8034515
39: -80.2488632, 30.4726276, -80.1966095, 30.4142628, -110.6631241, 110.6692352
40: -62.4555817, 25.5168610, -62.3884811, 25.3685818, -87.8241653, 87.9053421
41: -54.8758926, 32.7406120, -54.8176346, 32.6855125, -87.5614014, 87.5582428
42: -36.1823082, 25.8610420, -36.1615372, 25.9103432, -62.0926514, 62.0225792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1928414
time: 75.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
time: 110.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -56.4814453, 43.4717712, -56.5061493, 43.4465027, -99.9279480, 99.9779205
1: -25.2160568, 37.7067337, -25.2515240, 37.7333031, -62.9493599, 62.9582596
2: -21.7339706, 37.1441078, -21.7871094, 37.1988907, -58.9328613, 58.9312172
3: -24.3280163, 39.7356415, -24.4810333, 39.8509674, -64.1789856, 64.2166748
4: -28.3734303, 43.6627426, -28.4068432, 43.6909294, -72.0643616, 72.0695877
5: -24.5574913, 39.7027092, -24.6398506, 39.7341766, -64.2916718, 64.3425598
6: -54.2168770, 31.7651634, -54.2283707, 31.7884216, -86.0052948, 85.9935303
7: -30.3486404, 39.5429916, -30.3673496, 39.5011444, -69.8497849, 69.9103394
8: -36.5002975, 53.5011520, -36.5513268, 53.5629578, -90.0632553, 90.0524750
9: -29.0357876, 38.9778366, -29.0756168, 38.9918938, -68.0276794, 68.0534515
10: -49.3469391, 43.5193443, -49.4966049, 43.6637115, -93.0106506, 93.0159454
11: -48.8709641, 21.7351036, -49.0500565, 21.8779049, -70.7488708, 70.7851562
12: -55.1709824, 25.0050507, -55.1920471, 25.0923309, -79.1501846, 79.0425262
13: -50.3982086, 43.6465530, -50.5405922, 43.7608719, -94.1590805, 94.1871490
14: -87.2752075, 30.8949471, -87.3271027, 30.9561443, -118.2313538, 118.2220459
15: -35.6083527, 35.9764748, -35.7190857, 36.0163307, -71.6246796, 71.6955566
16: -45.7609406, 33.6975555, -45.8405190, 33.7764702, -79.5374146, 79.5380707
17: -84.6823807, 23.2914352, -84.7230911, 23.3426170, -108.0249939, 108.0145264
18: -48.8438187, 31.1733093, -49.0494080, 31.3288517, -80.1726685, 80.2227173
19: -38.8011436, 18.4080315, -38.9383812, 18.5429211, -57.3440628, 57.3464127
20: -36.8164520, 23.2455540, -36.8996773, 23.3196468, -60.1361008, 60.1452332
21: -47.9820747, 21.9273186, -48.1272888, 22.0606613, -70.0427399, 70.0546112
22: -49.9191170, 22.0460968, -49.8683586, 22.0318470, -71.9509659, 71.9144592
23: -38.9669113, 23.8652420, -39.0435829, 23.9450665, -62.9119797, 62.9088249
24: -46.1962280, 23.9043770, -46.2592430, 23.9604225, -70.1566467, 70.1636200
25: -41.1171303, 24.6622467, -41.1437225, 24.6716633, -65.7887955, 65.8059692
26: -56.8451042, 33.3267441, -56.9051437, 33.4777985, -90.3229065, 90.2318878
27: -45.2331924, 28.6831055, -45.2807274, 28.7164879, -73.9496765, 73.9638367
28: -38.8613167, 26.6315556, -38.9072723, 26.6801090, -65.5414276, 65.5388260
29: -51.5509567, 20.5189781, -51.5059090, 20.5006123, -72.0515671, 72.0248871
30: -49.1569138, 25.9998608, -49.2371750, 26.0769157, -75.2338257, 75.2370377
31: -50.8682327, 27.6883640, -51.0512085, 27.8315392, -78.6997681, 78.7395706
32: -52.3105927, 24.6060505, -52.3230095, 24.6443539, -76.9549484, 76.9290619
33: -72.1443405, 33.6572723, -72.2166443, 33.7466660, -105.6082230, 105.6464005
34: -65.4091263, 17.0539742, -65.4524994, 17.0992374, -81.6826630, 81.6959534
35: -63.6439552, 23.4411755, -63.7051888, 23.5312767, -85.6385040, 85.5753937
36: -61.8448524, 24.3104229, -61.8968239, 24.3651218, -86.2099762, 86.2072449
37: -87.0496445, 19.7322063, -87.0172882, 19.7241154, -106.7737579, 106.7494965
38: -69.7704086, 29.0536747, -69.8339081, 29.0985203, -98.8689270, 98.8875809
39: -80.2656250, 30.5094414, -80.2878647, 30.5270557, -110.7926788, 110.7973022
40: -62.4692764, 25.5746899, -62.4461098, 25.5266418, -87.9959183, 88.0207977
41: -54.8862305, 32.7676277, -54.8728142, 32.7677002, -87.6539307, 87.6404419
42: -36.1950760, 25.8737793, -36.2072983, 25.9618988, -62.1569748, 62.0810776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1306175
time: 94.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2363706
time: 95.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -56.2884064, 43.3686600, -56.4733620, 43.3767090, -99.6651154, 99.8420258
1: -25.0730286, 37.5942841, -25.2165279, 37.6781082, -62.7511368, 62.8108139
2: -21.5224094, 37.0090485, -21.7372360, 37.1514282, -58.6738358, 58.7462845
3: -24.0849762, 39.5826721, -24.3930779, 39.8181763, -63.9031525, 63.9757500
4: -28.1434689, 43.5222130, -28.3706722, 43.6292114, -71.7726822, 71.8928833
5: -24.3327065, 39.5676460, -24.5834885, 39.7071304, -64.0398407, 64.1511383
6: -54.1323624, 31.6666012, -54.2331009, 31.7559261, -85.8882904, 85.8997040
7: -30.1622238, 39.4554214, -30.3358154, 39.4533615, -69.6155853, 69.7912369
8: -36.3131371, 53.3421898, -36.4956207, 53.4930878, -89.8062286, 89.8378143
9: -28.9389992, 38.8919487, -29.0517673, 38.9532166, -67.8922119, 67.9437180
10: -49.2292328, 43.2957458, -49.5241928, 43.6793594, -92.9085922, 92.8199387
11: -48.6273575, 21.4483490, -49.0065079, 21.8171463, -70.4445038, 70.4548569
12: -55.0060043, 24.6938286, -55.1590958, 25.0275230, -78.9161224, 78.7149048
13: -50.2448425, 43.5045929, -50.4807205, 43.7547073, -93.9995499, 93.9853134
14: -87.0827789, 30.5638371, -87.3650131, 30.9750137, -118.0577927, 117.9288483
15: -35.4571724, 35.8970871, -35.6741028, 36.0096436, -71.4668121, 71.5711899
16: -45.6186752, 33.5496826, -45.8354378, 33.7084045, -79.3270798, 79.3851166
17: -84.4614105, 23.0332165, -84.7003937, 23.3197746, -107.7811890, 107.7336121
18: -48.6561127, 30.9621010, -49.0595245, 31.2270279, -79.8831406, 80.0216217
19: -38.6173782, 18.2201328, -38.8864670, 18.4604836, -57.0778618, 57.1065979
20: -36.6660614, 23.0743904, -36.8605118, 23.2817116, -59.9477730, 59.9349022
21: -47.7539139, 21.6931038, -48.0549431, 21.9753933, -69.7293091, 69.7480469
22: -49.7740784, 21.9037495, -49.7903671, 22.0080643, -71.7821426, 71.6941147
23: -38.8136597, 23.6742477, -39.0081444, 23.9053993, -62.7190590, 62.6823921
24: -46.0672684, 23.7684937, -46.2370758, 23.9416676, -70.0089340, 70.0055695
25: -41.0021667, 24.5075989, -41.1085281, 24.6650276, -65.6671906, 65.6161270
26: -56.6325874, 32.9887886, -56.8578491, 33.3820724, -90.0146637, 89.8466339
27: -45.0748520, 28.5727787, -45.2730370, 28.6846008, -73.7594528, 73.8458176
28: -38.7034683, 26.4376793, -38.8647308, 26.6328850, -65.3363495, 65.3024139
29: -51.3792610, 20.3344688, -51.4150238, 20.4745293, -71.8537903, 71.7494965
30: -48.9857101, 25.7622604, -49.1831779, 26.0433140, -75.0290222, 74.9454346
31: -50.6629829, 27.4839573, -51.0124855, 27.7696209, -78.4326019, 78.4964447
32: -52.2068138, 24.5039558, -52.2979355, 24.6190033, -76.8258209, 76.8018951
33: -71.9488678, 33.5289268, -72.2035675, 33.7651634, -105.4019775, 105.5056076
34: -65.3220520, 16.9544697, -65.4331818, 17.0756245, -81.4861679, 81.5929337
35: -63.5607986, 23.3696537, -63.6709976, 23.5167065, -85.5179749, 85.5412064
36: -61.7692375, 24.2418022, -61.8772583, 24.3479252, -86.1171646, 86.1190643
37: -86.9465179, 19.6254292, -87.0845032, 19.7340374, -106.6805573, 106.7099304
38: -69.6556091, 28.9865551, -69.8537903, 29.1106586, -98.7662659, 98.8403473
39: -80.1130447, 30.4142742, -80.3012466, 30.5312176, -110.6442642, 110.7155228
40: -62.3679657, 25.4942875, -62.4832077, 25.5161533, -87.8841171, 87.9774933
41: -54.8031998, 32.6783981, -54.9129372, 32.7646332, -87.5678329, 87.5913391
42: -36.1168060, 25.7586040, -36.1679459, 25.9330978, -62.0499039, 61.9265518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1346312
time: 91.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2302435
time: 77.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -56.3056641, 43.4187775, -56.6037140, 43.5313759, -99.8370361, 100.0224915
1: -25.0826836, 37.6441422, -25.3123779, 37.8234673, -62.9061508, 62.9565201
2: -21.5315666, 37.0463257, -21.8392544, 37.2606583, -58.7922249, 58.8855820
3: -24.0912247, 39.6077080, -24.4568634, 39.8972931, -63.9885178, 64.0645752
4: -28.1545620, 43.5870628, -28.5112991, 43.8240967, -71.9786606, 72.0983582
5: -24.3413315, 39.6085968, -24.6713181, 39.8319206, -64.1732483, 64.2799149
6: -54.1415062, 31.6939087, -54.2759705, 31.8506107, -85.9921188, 85.9698792
7: -30.1752090, 39.5161514, -30.4588890, 39.6289711, -69.8041840, 69.9750366
8: -36.3217354, 53.3994217, -36.6045341, 53.6605148, -89.9822540, 90.0039520
9: -28.9460983, 38.9154205, -29.1023846, 39.0331650, -67.9792633, 68.0178070
10: -49.2515259, 43.3119812, -49.6050034, 43.7703781, -93.0219040, 92.9169846
11: -48.6813469, 21.4574852, -49.1740036, 21.9127254, -70.5940704, 70.6314850
12: -55.0686951, 24.7041855, -55.3461990, 25.1673298, -79.1248474, 78.9016418
13: -50.2674561, 43.5174904, -50.5680618, 43.8049088, -94.0723648, 94.0855560
14: -87.1464233, 30.5712662, -87.5766907, 31.1142597, -118.2606812, 118.1479568
15: -35.4811401, 35.9065666, -35.7802048, 36.0475082, -71.5286484, 71.6867676
16: -45.6359520, 33.5985489, -45.9141388, 33.8669281, -79.5028839, 79.5126877
17: -84.5453644, 23.0436497, -84.9580383, 23.4395351, -107.9849014, 108.0016861
18: -48.6720200, 30.9789028, -49.0731659, 31.2634964, -79.9355164, 80.0520706
19: -38.6484108, 18.2244263, -38.9918671, 18.5124054, -57.1608162, 57.2162933
20: -36.6935158, 23.0797348, -36.9594421, 23.3459320, -60.0394478, 60.0391769
21: -47.8032036, 21.6992569, -48.2153969, 22.0612259, -69.8644257, 69.9146576
22: -49.8588905, 21.9104538, -50.0452576, 22.1102676, -71.9691620, 71.9557114
23: -38.8580360, 23.6799088, -39.1484528, 23.9916592, -62.8496933, 62.8283615
24: -46.1045876, 23.7733459, -46.3574333, 23.9967079, -70.1012955, 70.1307831
25: -41.0562592, 24.5164948, -41.2732086, 24.7690125, -65.8252716, 65.7897034
26: -56.6893883, 32.9961739, -57.0416641, 33.5029984, -90.1923828, 90.0378418
27: -45.0901299, 28.5864716, -45.3281784, 28.7265816, -73.8167114, 73.9146500
28: -38.7387466, 26.4429760, -38.9834251, 26.7157898, -65.4545364, 65.4263992
29: -51.4784088, 20.3402328, -51.7128105, 20.5990124, -72.0774231, 72.0530396
30: -49.0403214, 25.7720490, -49.3493881, 26.1519814, -75.1923065, 75.1214371
31: -50.6935806, 27.4901543, -51.1162338, 27.8311825, -78.5247650, 78.6063843
32: -52.2298584, 24.5113182, -52.3805161, 24.6600571, -76.8899155, 76.8918304
33: -71.9631882, 33.5414963, -72.2782288, 33.8191986, -105.4735794, 105.7131882
34: -65.3364334, 16.9663849, -65.4903412, 17.1366024, -81.5544128, 81.7383957
35: -63.5848541, 23.3798637, -63.7504959, 23.5540409, -85.5621490, 85.6384125
36: -61.7910690, 24.2502861, -61.9539490, 24.3912315, -86.1822968, 86.2042389
37: -86.9649963, 19.6445618, -87.1413498, 19.8028183, -106.7678146, 106.7859116
38: -69.6714554, 29.0155983, -69.9130249, 29.2008667, -98.8723221, 98.9286194
39: -80.1304092, 30.4510574, -80.3959961, 30.6448746, -110.7752838, 110.8470535
40: -62.3816719, 25.5587711, -62.5538025, 25.6919193, -88.0735931, 88.1125717
41: -54.8133202, 32.7048187, -54.9736862, 32.8565979, -87.6699219, 87.6785049
42: -36.1284103, 25.7719975, -36.2150269, 25.9892006, -62.1176109, 61.9870224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1204421
time: 97.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2340991
time: 112.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -56.4771156, 43.4461098, -56.5283813, 43.3908386, -99.8679504, 99.9744873
1: -25.2138920, 37.6861610, -25.2595119, 37.6871490, -62.9010391, 62.9456711
2: -21.7330875, 37.1281548, -21.8060226, 37.1616745, -58.8947601, 58.9341774
3: -24.3256798, 39.7223969, -24.4720612, 39.8305779, -64.1562576, 64.1944580
4: -28.3738365, 43.6419678, -28.4457741, 43.6396637, -72.0135040, 72.0877380
5: -24.5553055, 39.6900711, -24.6552677, 39.7194481, -64.2747498, 64.3453369
6: -54.2139587, 31.7571964, -54.2516899, 31.7825508, -85.9965057, 86.0088882
7: -30.3461380, 39.5214195, -30.3914280, 39.4621811, -69.8083191, 69.9128494
8: -36.4980965, 53.4747620, -36.5544891, 53.5076828, -90.0057831, 90.0292511
9: -29.0350094, 38.9639206, -29.0805950, 38.9718475, -68.0068588, 68.0445175
10: -49.3593254, 43.5128059, -49.5499268, 43.7455139, -93.1048431, 93.0627289
11: -48.8577957, 21.7327309, -49.0282593, 21.9130592, -70.7708588, 70.7609863
12: -55.1553612, 25.0009460, -55.1736450, 25.1281357, -79.1680450, 79.0156937
13: -50.3887405, 43.6437111, -50.5304489, 43.7811279, -94.1698685, 94.1741638
14: -87.2844696, 30.8922176, -87.3950272, 31.0866203, -118.3710938, 118.2872467
15: -35.6044579, 35.9742355, -35.7164192, 36.0270462, -71.6315002, 71.6906586
16: -45.7560081, 33.6759224, -45.8701515, 33.7446289, -79.5006409, 79.5460739
17: -84.6689835, 23.2880878, -84.7197418, 23.4051094, -108.0740967, 108.0078278
18: -48.8436356, 31.1690006, -49.0762596, 31.3038445, -80.1474762, 80.2452621
19: -38.7839661, 18.4054718, -38.9068680, 18.5224705, -57.3064346, 57.3123398
20: -36.8061676, 23.2449379, -36.8822632, 23.3386021, -60.1447678, 60.1272011
21: -47.9578629, 21.9251766, -48.0779114, 22.0535202, -70.0113831, 70.0030899
22: -49.8861122, 22.0455418, -49.8092308, 22.0534687, -71.9395828, 71.8547745
23: -38.9539490, 23.8637085, -39.0256233, 23.9679871, -62.9219360, 62.8893318
24: -46.1860733, 23.9032745, -46.2560577, 23.9861069, -70.1721802, 70.1593323
25: -41.1034088, 24.6640987, -41.1261444, 24.7166748, -65.8200836, 65.7902451
26: -56.8268394, 33.3229218, -56.8799553, 33.4935265, -90.3203659, 90.2028809
27: -45.2275810, 28.6827774, -45.2961998, 28.7269726, -73.9545517, 73.9789734
28: -38.8478928, 26.6308765, -38.8851471, 26.6969261, -65.5448151, 65.5160217
29: -51.5140877, 20.5190201, -51.4315948, 20.5359612, -72.0500488, 71.9506149
30: -49.1405792, 25.9985161, -49.2019463, 26.1206741, -75.2612534, 75.2004623
31: -50.8567810, 27.6860046, -51.0367699, 27.8369293, -78.6937103, 78.7227783
32: -52.3010368, 24.6040230, -52.3186531, 24.6487846, -76.9498215, 76.9226761
33: -72.1409149, 33.6688004, -72.2655487, 33.7870636, -105.5984268, 105.7163162
34: -65.4028473, 17.0510902, -65.4556580, 17.0994949, -81.6062469, 81.7218246
35: -63.6279907, 23.4379292, -63.6902771, 23.5308495, -85.6143036, 85.6547928
36: -61.8319435, 24.3074951, -61.8922462, 24.3642578, -86.1961975, 86.1997375
37: -87.0456009, 19.7353764, -87.1056595, 19.7695675, -106.8151703, 106.8410339
38: -69.7651062, 29.0548687, -69.8825760, 29.1251183, -98.8902283, 98.9374466
39: -80.2604752, 30.5122471, -80.3470383, 30.5461388, -110.8066101, 110.8592834
40: -62.4651070, 25.5679665, -62.5056000, 25.5313034, -87.9964142, 88.0735626
41: -54.8840408, 32.7695503, -54.9342422, 32.7900734, -87.6741180, 87.7037964
42: -36.1827545, 25.8730888, -36.1836662, 25.9642754, -62.1470299, 62.0567551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1405458
time: 76.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
time: 92.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -56.4944000, 43.4970703, -56.6589241, 43.5452423, -100.0396423, 100.1559906
1: -25.2234402, 37.7359695, -25.3552704, 37.8323708, -63.0558090, 63.0912399
2: -21.7422104, 37.1652946, -21.9076691, 37.2706375, -59.0128479, 59.0729637
3: -24.3318119, 39.7473297, -24.5353432, 39.9093513, -64.2411652, 64.2826691
4: -28.3846874, 43.7087555, -28.5858288, 43.8344955, -72.2191849, 72.2945862
5: -24.5637817, 39.7310181, -24.7425652, 39.8440247, -64.4078064, 64.4735870
6: -54.2232666, 31.7841854, -54.2943039, 31.8768845, -86.1001511, 86.0784912
7: -30.3589325, 39.5816116, -30.5140762, 39.6374664, -69.9963989, 70.0956879
8: -36.5067215, 53.5318451, -36.6631012, 53.6748047, -90.1815262, 90.1949463
9: -29.0423126, 38.9872589, -29.1319771, 39.0507050, -68.0930176, 68.1192322
10: -49.3817596, 43.5287666, -49.6306572, 43.8357506, -93.2175140, 93.1594238
11: -48.9116821, 21.7416382, -49.1951675, 22.0085106, -70.9201965, 70.9368057
12: -55.2180367, 25.0110855, -55.3607216, 25.2669239, -79.3772736, 79.2143707
13: -50.4066086, 43.6568146, -50.6113434, 43.8312912, -94.2378998, 94.2681580
14: -87.3502960, 30.8997154, -87.6067276, 31.2248974, -118.5751953, 118.5064392
15: -35.6264496, 35.9836235, -35.8231125, 36.0646820, -71.6911316, 71.8067322
16: -45.7732811, 33.7270775, -45.9485168, 33.9053612, -79.6786423, 79.6755981
17: -84.7528687, 23.2983513, -84.9771271, 23.5242729, -108.2771454, 108.2754822
18: -48.8594780, 31.1684818, -49.0892868, 31.3273163, -80.1867981, 80.2577667
19: -38.8147774, 18.4098396, -39.0117874, 18.5742645, -57.3890419, 57.4216270
20: -36.8335838, 23.2502022, -36.9809647, 23.4025726, -60.2361565, 60.2311668
21: -48.0070190, 21.9315777, -48.2380829, 22.1392918, -70.1463089, 70.1696625
22: -49.9722214, 22.0521755, -50.0634995, 22.1553078, -72.1275330, 72.1156769
23: -38.9985580, 23.8694210, -39.1656952, 24.0540276, -63.0525856, 63.0351181
24: -46.2229424, 23.9081955, -46.3756409, 24.0413284, -70.2642670, 70.2838364
25: -41.1568451, 24.6728802, -41.2900810, 24.8205471, -65.9773941, 65.9629593
26: -56.8831177, 33.3303795, -57.0634537, 33.6139526, -90.4970703, 90.3938293
27: -45.2429695, 28.6877079, -45.3508759, 28.7573395, -74.0003052, 74.0385818
28: -38.8831406, 26.6360722, -39.0035591, 26.7796593, -65.6627960, 65.6396332
29: -51.6131706, 20.5247726, -51.7293320, 20.6600552, -72.2732239, 72.2541046
30: -49.1949883, 26.0079021, -49.3676682, 26.2292385, -75.4242249, 75.3755722
31: -50.8872261, 27.6923199, -51.1400948, 27.8988495, -78.7860718, 78.8324127
32: -52.3249397, 24.6109676, -52.4009476, 24.6891785, -77.0141144, 77.0119171
33: -72.1548615, 33.6812973, -72.3399429, 33.8410263, -105.6888733, 105.9259720
34: -65.4175262, 17.0629463, -65.5122757, 17.1608868, -81.6819305, 81.8656464
35: -63.6523666, 23.4483070, -63.7696266, 23.5680408, -85.6580353, 85.7484360
36: -61.8536987, 24.3160381, -61.9684982, 24.4074860, -86.2611847, 86.2845383
37: -87.0641251, 19.7551079, -87.1621246, 19.8359947, -106.9001160, 106.9172363
38: -69.7807617, 29.0845966, -69.9416046, 29.2155781, -98.9963379, 99.0261993
39: -80.2774124, 30.5490360, -80.4414368, 30.6596375, -110.9370499, 110.9904709
40: -62.4790268, 25.6269894, -62.5761948, 25.7058144, -88.1848450, 88.2031860
41: -54.8944016, 32.7973289, -54.9948540, 32.8816910, -87.7760925, 87.7921829
42: -36.1958771, 25.8860092, -36.2305222, 26.0209484, -62.2168274, 62.1165314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1307067
time: 74.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1307067
time: 110.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -56.4853172, 43.4051208, -56.3800430, 43.2988701, -99.7841873, 99.7851639
1: -25.2389145, 37.6745148, -25.1652164, 37.5923767, -62.8312912, 62.8397293
2: -21.7334747, 37.1069870, -21.6877060, 37.0935669, -58.8270416, 58.7946930
3: -24.3524055, 39.7478371, -24.4276886, 39.7796707, -64.1320801, 64.1755219
4: -28.3987770, 43.6218643, -28.2790565, 43.4999046, -71.8986816, 71.9009247
5: -24.5632133, 39.6691017, -24.5559120, 39.6155434, -64.1787567, 64.2250137
6: -54.2153969, 31.7234802, -54.1937866, 31.6934471, -85.9088440, 85.9172668
7: -30.3533840, 39.4866295, -30.2491550, 39.3313408, -69.6847229, 69.7357864
8: -36.5128403, 53.4745255, -36.4493446, 53.4036102, -89.9164505, 89.9238739
9: -29.0401649, 38.9739304, -29.0268192, 38.9200974, -67.9602661, 68.0007477
10: -49.4756508, 43.6986694, -49.4255981, 43.6409073, -93.1165619, 93.1242676
11: -48.8959885, 21.8107948, -48.8906860, 21.8127060, -70.7086945, 70.7014771
12: -55.1224251, 25.0308533, -55.0115128, 24.9638023, -78.9664993, 78.9051590
13: -50.4673691, 43.6963081, -50.4870148, 43.7165337, -94.1838989, 94.1833191
14: -87.2990112, 31.0077305, -87.1268311, 30.8571796, -118.1561890, 118.1345596
15: -35.6372757, 35.9892502, -35.6386681, 35.9869156, -71.6241913, 71.6279144
16: -45.8147049, 33.7078209, -45.7758904, 33.6408844, -79.4555893, 79.4837112
17: -84.6388397, 23.3295555, -84.4711914, 23.2391472, -107.8779907, 107.8007507
18: -48.8866348, 31.2449627, -49.0447769, 31.3252869, -80.2119217, 80.2897415
19: -38.8224869, 18.4387589, -38.8418655, 18.5029144, -57.3254013, 57.2806244
20: -36.8190079, 23.2745819, -36.8069763, 23.2662125, -60.0852203, 60.0815582
21: -47.9839668, 21.9688454, -47.9746666, 21.9897881, -69.9737549, 69.9435120
22: -49.8366852, 22.0299835, -49.6228714, 21.9261894, -71.7628784, 71.6528549
23: -38.9700775, 23.9173393, -38.9100800, 23.8767052, -62.8467827, 62.8274193
24: -46.2147255, 23.9434700, -46.1481895, 23.9206200, -70.1353455, 70.0916595
25: -41.1174698, 24.6956177, -40.9881592, 24.5823097, -65.6997833, 65.6837769
26: -56.7970352, 33.3619690, -56.7288628, 33.3686180, -90.1656494, 90.0908356
27: -45.1944733, 28.6517868, -45.2343826, 28.6764412, -73.8709106, 73.8861694
28: -38.8269196, 26.6213989, -38.7946167, 26.5943661, -65.4212875, 65.4160156
29: -51.4572754, 20.5075760, -51.2155113, 20.3735123, -71.8307877, 71.7230835
30: -49.1441116, 26.0494480, -49.0788841, 25.9912395, -75.1353531, 75.1283340
31: -50.9253044, 27.7608566, -50.9588318, 27.7974072, -78.7227097, 78.7196884
32: -52.2914467, 24.5809479, -52.2514610, 24.5975151, -76.8889618, 76.8324127
33: -72.1951599, 33.6829910, -72.1627502, 33.6933517, -105.6025238, 105.4707108
34: -65.4373322, 17.0593853, -65.4125671, 17.0431900, -81.6682587, 81.5888672
35: -63.6993179, 23.4958172, -63.6536636, 23.4985123, -85.6960449, 85.5720673
36: -61.9096184, 24.3440304, -61.8508492, 24.3227692, -86.2323914, 86.1948776
37: -87.0535278, 19.7043514, -86.9787064, 19.6580677, -106.7115936, 106.6830597
38: -69.8505020, 29.1068611, -69.8091736, 29.0240059, -98.8745117, 98.9160309
39: -80.3160477, 30.5051289, -80.2150192, 30.4161968, -110.7322464, 110.7201462
40: -62.4813156, 25.5208931, -62.3980865, 25.3657303, -87.8470459, 87.9189758
41: -54.9240036, 32.7406387, -54.8297043, 32.6828728, -87.6068726, 87.5703430
42: -36.1856384, 25.8721638, -36.1641617, 25.9115677, -62.0972061, 62.0363235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1898741
time: 93.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
time: 91.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -56.5022736, 43.4540787, -56.5084572, 43.4484711, -99.9507446, 99.9625397
1: -25.2485161, 37.7243309, -25.2602139, 37.7371674, -62.9856834, 62.9845428
2: -21.7424850, 37.1441956, -21.7884064, 37.2023468, -58.9448318, 58.9326019
3: -24.3585548, 39.7728310, -24.4903316, 39.8578224, -64.2163773, 64.2631607
4: -28.4097633, 43.6866264, -28.4179058, 43.6939468, -72.1037140, 72.1045303
5: -24.5717144, 39.7099380, -24.6426964, 39.7396278, -64.3113403, 64.3526306
6: -54.2245102, 31.7494164, -54.2345085, 31.7822933, -86.0068054, 85.9839249
7: -30.3662758, 39.5470619, -30.3709774, 39.5057220, -69.8719940, 69.9180374
8: -36.5213318, 53.5316544, -36.5570297, 53.5698891, -90.0912170, 90.0886841
9: -29.0472679, 38.9972420, -29.0757771, 38.9977303, -68.0449982, 68.0730209
10: -49.4978828, 43.7145844, -49.5055618, 43.7291222, -93.2270050, 93.2201462
11: -48.9498520, 21.8197498, -49.0570908, 21.9067497, -70.8565979, 70.8768387
12: -55.1849899, 25.0410595, -55.1975250, 25.1023369, -79.1736679, 79.0904541
13: -50.4897308, 43.7094231, -50.5714111, 43.7656860, -94.2554169, 94.2808380
14: -87.3625031, 31.0151138, -87.3369446, 30.9947052, -118.3572083, 118.3520584
15: -35.6601486, 35.9987183, -35.7367706, 36.0228882, -71.6830368, 71.7354889
16: -45.8319321, 33.7560349, -45.8531609, 33.7925568, -79.6244888, 79.6091919
17: -84.7227631, 23.3397179, -84.7278595, 23.3564987, -108.0792618, 108.0675812
18: -48.9029083, 31.2574863, -49.0565033, 31.3596916, -80.2626038, 80.3139877
19: -38.8534317, 18.4429359, -38.9460983, 18.5540123, -57.4074440, 57.3890343
20: -36.8463516, 23.2797775, -36.9048615, 23.3299294, -60.1762810, 60.1846390
21: -48.0331230, 21.9749012, -48.1335335, 22.0750008, -70.1081238, 70.1084366
22: -49.9208603, 22.0366459, -49.8743591, 22.0272408, -71.9481049, 71.9110031
23: -39.0144196, 23.9229069, -39.0498161, 23.9620819, -62.9765015, 62.9727249
24: -46.2515678, 23.9483185, -46.2669106, 23.9750481, -70.2266159, 70.2152252
25: -41.1713295, 24.7044678, -41.1518707, 24.6845322, -65.8558655, 65.8563385
26: -56.8537216, 33.3691940, -56.9114571, 33.4883690, -90.3420868, 90.2806549
27: -45.2095680, 28.6653061, -45.2873535, 28.7124081, -73.9219742, 73.9526596
28: -38.8621559, 26.6266384, -38.9127426, 26.6764088, -65.5385666, 65.5393829
29: -51.5556526, 20.5133209, -51.5102539, 20.4969635, -72.0526123, 72.0235748
30: -49.1984787, 26.0591888, -49.2437973, 26.0987701, -75.2972488, 75.3029861
31: -50.9558640, 27.7669411, -51.0615768, 27.8581200, -78.8139801, 78.8285217
32: -52.3137093, 24.5882607, -52.3280373, 24.6377182, -76.9514313, 76.9162979
33: -72.2094116, 33.6954727, -72.2357864, 33.7467117, -105.6731567, 105.6726227
34: -65.4513321, 17.0712624, -65.4660721, 17.1016216, -81.7325287, 81.7113876
35: -63.7230644, 23.5060291, -63.7306747, 23.5353069, -85.7344894, 85.6598816
36: -61.9306107, 24.3526211, -61.9245071, 24.3656788, -86.2962875, 86.2771301
37: -87.0718002, 19.7224483, -87.0285187, 19.7196159, -106.7914124, 106.7509689
38: -69.8662109, 29.1358891, -69.8649826, 29.1056404, -98.9718475, 99.0008698
39: -80.3329620, 30.5418663, -80.3069458, 30.5291557, -110.8621216, 110.8488159
40: -62.4948006, 25.5790501, -62.4560471, 25.5229950, -88.0177917, 88.0350952
41: -54.9340744, 32.7661095, -54.8853226, 32.7655602, -87.6996307, 87.6514282
42: -36.1971817, 25.8855896, -36.2101517, 25.9626751, -62.1598587, 62.0957413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1217664
time: 105.73 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2352701
time: 88.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -56.6757469, 43.4835892, -56.4353485, 43.3131828, -99.9889297, 99.9189377
1: -25.3816280, 37.7677345, -25.2085514, 37.6016769, -62.9833069, 62.9762878
2: -21.9460793, 37.2275925, -21.7567539, 37.1040688, -59.0501480, 58.9843445
3: -24.5950756, 39.8898544, -24.5069218, 39.7923355, -64.3874130, 64.3967743
4: -28.6318741, 43.7436905, -28.3544693, 43.5105896, -72.1424637, 72.0981598
5: -24.7874680, 39.7929115, -24.6279392, 39.6280212, -64.4154892, 64.4208527
6: -54.2996788, 31.8172054, -54.2124100, 31.7203674, -86.0200500, 86.0296173
7: -30.5401783, 39.5528946, -30.3050823, 39.3404427, -69.8806229, 69.8579788
8: -36.7003174, 53.6085854, -36.5086212, 53.4184418, -90.1187592, 90.1172028
9: -29.1400127, 39.0487251, -29.0560398, 38.9389725, -68.0789871, 68.1047668
10: -49.6103630, 43.9191437, -49.4519196, 43.7075996, -93.3179626, 93.3710632
11: -49.1312714, 22.0979519, -48.9129486, 21.9088650, -71.0401382, 71.0109024
12: -55.2720909, 25.3401928, -55.0262909, 25.0647888, -79.2189789, 79.2185211
13: -50.6199989, 43.8351288, -50.5381470, 43.7430763, -94.3630753, 94.3732758
14: -87.5032959, 31.3381062, -87.1573105, 30.9687634, -118.4720612, 118.4954147
15: -35.8019104, 36.0685196, -35.6872368, 36.0046501, -71.8065643, 71.7557526
16: -45.9588966, 33.8362274, -45.8113785, 33.6780472, -79.6369476, 79.6476059
17: -84.8476486, 23.5880394, -84.4910126, 23.3246346, -108.1722870, 108.0790558
18: -49.0765762, 31.4627018, -49.0618439, 31.4027100, -80.4792862, 80.5245438
19: -38.9924736, 18.6256618, -38.8626404, 18.5651779, -57.5576515, 57.4883041
20: -36.9606285, 23.4460983, -36.8289223, 23.3232040, -60.2838326, 60.2750206
21: -48.1909218, 22.2022400, -47.9979324, 22.0681152, -70.2590332, 70.2001724
22: -49.9487686, 22.1759262, -49.6423264, 21.9719067, -71.9206772, 71.8182526
23: -39.1123810, 24.1087132, -38.9278870, 23.9394531, -63.0518341, 63.0365982
24: -46.3348732, 24.0790520, -46.1675415, 23.9650650, -70.2999420, 70.2465973
25: -41.2208138, 24.8540478, -41.0062790, 24.6339226, -65.8547363, 65.8603287
26: -56.9932976, 33.7012634, -56.7514992, 33.4805641, -90.4738617, 90.4527588
27: -45.3505402, 28.7746029, -45.2576942, 28.7251434, -74.0756836, 74.0322952
28: -38.9725494, 26.8165379, -38.8152924, 26.6586227, -65.6311722, 65.6318283
29: -51.5927048, 20.6940346, -51.2323990, 20.4349766, -72.0276794, 71.9264374
30: -49.3024139, 26.2881870, -49.0980530, 26.0688095, -75.3712234, 75.3862381
31: -51.1222992, 27.9643669, -50.9835052, 27.8648968, -78.9871979, 78.9478760
32: -52.3865051, 24.6839256, -52.2726440, 24.6277866, -77.0142899, 76.9565735
33: -72.3889465, 33.8253059, -72.2246628, 33.7155533, -105.8208466, 105.6840210
34: -65.5210266, 17.1611671, -65.4357910, 17.0679779, -81.7932510, 81.7271194
35: -63.7680206, 23.5662346, -63.6730347, 23.5128784, -85.8014374, 85.6786957
36: -61.9746323, 24.4110432, -61.8659515, 24.3392258, -86.3138580, 86.2769928
37: -87.1563110, 19.8225212, -86.9995880, 19.6950951, -106.8514099, 106.8221130
38: -69.9643707, 29.1784515, -69.8384628, 29.0396423, -99.0040131, 99.0169144
39: -80.4694443, 30.6047401, -80.2612305, 30.4313221, -110.9007645, 110.8659668
40: -62.5807610, 25.5929813, -62.4202003, 25.3793354, -87.9600983, 88.0131836
41: -55.0080070, 32.8326187, -54.8510437, 32.7075043, -87.7155151, 87.6836624
42: -36.2540512, 25.9927063, -36.1800652, 25.9432526, -62.1973038, 62.1727715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0636384, upper bound: 44.1950421
time: 74.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0636384, upper bound: 44.2366513
time: 91.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -56.6930580, 43.5334320, -56.5640411, 43.4625206, -100.1555786, 100.0974731
1: -25.3912468, 37.8175964, -25.3035240, 37.7462540, -63.1375008, 63.1211205
2: -21.9552841, 37.2647858, -21.8572731, 37.2125702, -59.1678543, 59.1220589
3: -24.6014805, 39.9148712, -24.5691872, 39.8701935, -64.4716721, 64.4840546
4: -28.6431332, 43.8105011, -28.4929447, 43.7046318, -72.3477631, 72.3034439
5: -24.7959938, 39.8338852, -24.7143517, 39.7519379, -64.5479279, 64.5482330
6: -54.3091927, 31.8449459, -54.2530174, 31.8089294, -86.1181183, 86.0979614
7: -30.5530128, 39.6129684, -30.4266682, 39.5143623, -70.0673752, 70.0396347
8: -36.7092285, 53.6657333, -36.6161346, 53.5844269, -90.2936554, 90.2818680
9: -29.1480026, 39.0719986, -29.1059551, 39.0161285, -68.1641312, 68.1779556
10: -49.6330261, 43.9352036, -49.5318794, 43.7952538, -93.4282837, 93.4670868
11: -49.1854019, 22.1071091, -49.0787735, 22.0028076, -71.1882095, 71.1858826
12: -55.3347397, 25.3504715, -55.2121277, 25.2025986, -79.4269257, 79.4067535
13: -50.6364517, 43.8487320, -50.6150246, 43.7922745, -94.4287262, 94.4637604
14: -87.5691910, 31.3457432, -87.3673172, 31.1055660, -118.6747589, 118.7130585
15: -35.8251305, 36.0780907, -35.7841301, 36.0404282, -71.8655548, 71.8622208
16: -45.9766006, 33.8861923, -45.8885345, 33.8313599, -79.8079605, 79.7747269
17: -84.9315643, 23.5985031, -84.7472076, 23.4416542, -108.3732147, 108.3457108
18: -49.0928650, 31.4621410, -49.0729637, 31.4249306, -80.5177917, 80.5351028
19: -39.0235062, 18.6300888, -38.9664459, 18.6161366, -57.6396408, 57.5965347
20: -36.9881363, 23.4513588, -36.9266739, 23.3867340, -60.3748703, 60.3780327
21: -48.2402191, 22.2086163, -48.1566238, 22.1533012, -70.3935242, 70.3652420
22: -50.0346603, 22.1826973, -49.8929634, 22.0728168, -72.1074753, 72.0756607
23: -39.1571045, 24.1145668, -39.0674095, 24.0246830, -63.1817856, 63.1819763
24: -46.3715782, 24.0840874, -46.2854424, 24.0197144, -70.3912964, 70.3695297
25: -41.2742691, 24.8630142, -41.1691322, 24.7362709, -66.0105438, 66.0321503
26: -57.0496178, 33.7088547, -56.9335327, 33.6001167, -90.6497345, 90.6423874
27: -45.3661194, 28.7788181, -45.3103142, 28.7476196, -74.1137390, 74.0891342
28: -39.0079231, 26.8218536, -38.9331589, 26.7405796, -65.7485046, 65.7550125
29: -51.6913986, 20.6999264, -51.5269508, 20.5582428, -72.2496414, 72.2268753
30: -49.3569374, 26.2978668, -49.2624779, 26.1761742, -75.5331116, 75.5603485
31: -51.1530914, 27.9707451, -51.0858955, 27.9259968, -79.0790863, 79.0566406
32: -52.4094009, 24.6910095, -52.3490677, 24.6673279, -77.0767288, 77.0400772
33: -72.4031677, 33.8378716, -72.2976837, 33.7687531, -105.8915405, 105.8884277
34: -65.5360107, 17.1731319, -65.4884796, 17.1268349, -81.8586121, 81.8454514
35: -63.7922668, 23.5767765, -63.7498398, 23.5495682, -85.8402863, 85.7739487
36: -61.9960327, 24.4197636, -61.9390869, 24.3821735, -86.3782043, 86.3588486
37: -87.1750183, 19.8401146, -87.0492096, 19.7540283, -106.9290466, 106.8893280
38: -69.9804230, 29.2082729, -69.8938904, 29.1212463, -99.1016693, 99.1021652
39: -80.4864044, 30.6416607, -80.3530426, 30.5440636, -111.0304718, 110.9947052
40: -62.5952110, 25.6501198, -62.4783173, 25.5362511, -88.1314621, 88.1284332
41: -55.0185471, 32.8586540, -54.9066544, 32.7897415, -87.8082886, 87.7653046
42: -36.2672043, 26.0073071, -36.2258644, 25.9950905, -62.2622948, 62.2331696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1312242
time: 72.31 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2379575
time: 97.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -56.4982224, 43.4292641, -56.5309181, 43.3922577, -99.8904800, 99.9601822
1: -25.2461281, 37.7036514, -25.2682915, 37.6906586, -62.9367867, 62.9719429
2: -21.7416248, 37.1268578, -21.8069305, 37.1649132, -58.9065399, 58.9337883
3: -24.3561745, 39.7597694, -24.4820957, 39.8371582, -64.1933289, 64.2418671
4: -28.4102097, 43.6642761, -28.4547253, 43.6428528, -72.0530624, 72.1190033
5: -24.5695477, 39.6975708, -24.6584549, 39.7248383, -64.2943878, 64.3560257
6: -54.2218246, 31.7418823, -54.2582588, 31.7772217, -85.9990463, 86.0001373
7: -30.3636360, 39.5248070, -30.3950291, 39.4657135, -69.8293457, 69.9198380
8: -36.5190849, 53.5050240, -36.5602646, 53.5141373, -90.0332184, 90.0652924
9: -29.0466366, 38.9833908, -29.0825787, 38.9778214, -68.0244598, 68.0659714
10: -49.5101585, 43.7075119, -49.5593681, 43.8123856, -93.3225403, 93.2668762
11: -48.9345970, 21.8172264, -49.0348473, 21.9430466, -70.8776398, 70.8520737
12: -55.1692810, 25.0369987, -55.1787949, 25.1387024, -79.1938629, 79.0642242
13: -50.4750366, 43.7074661, -50.5553322, 43.7860107, -94.2610474, 94.2628021
14: -87.3744049, 31.0125675, -87.4048615, 31.1272182, -118.5016251, 118.4174271
15: -35.6508789, 35.9962463, -35.7332497, 36.0335999, -71.6844788, 71.7294922
16: -45.8268242, 33.7329330, -45.8831482, 33.7634659, -79.5902863, 79.6160812
17: -84.7098160, 23.3361969, -84.7239838, 23.4197216, -108.1295395, 108.0601807
18: -48.9034691, 31.2314968, -49.0828705, 31.3190536, -80.2225189, 80.3143692
19: -38.8360214, 18.4403191, -38.9144211, 18.5338345, -57.3698578, 57.3547401
20: -36.8352890, 23.2792740, -36.8875008, 23.3492813, -60.1845703, 60.1667747
21: -48.0087242, 21.9730492, -48.0842781, 22.0685253, -70.0772476, 70.0573273
22: -49.8889847, 22.0359993, -49.8142548, 22.0498543, -71.9388428, 71.8502502
23: -39.0017548, 23.9214859, -39.0319252, 23.9859905, -62.9877472, 62.9534111
24: -46.2415123, 23.9473209, -46.2630692, 24.0017433, -70.2432556, 70.2103882
25: -41.1569214, 24.7064629, -41.1332779, 24.7304344, -65.8873596, 65.8397369
26: -56.8351479, 33.3652496, -56.8852730, 33.5048141, -90.3399658, 90.2505188
27: -45.2042198, 28.6572132, -45.3031502, 28.7132549, -73.9174728, 73.9603653
28: -38.8480988, 26.6258640, -38.8906670, 26.6935463, -65.5416412, 65.5165329
29: -51.5184555, 20.5135460, -51.4354172, 20.5326328, -72.0510864, 71.9489594
30: -49.1791992, 26.0575256, -49.2081070, 26.1386986, -75.3179016, 75.2656326
31: -50.9442673, 27.7646675, -51.0471954, 27.8646011, -78.8088684, 78.8118591
32: -52.3032303, 24.5859985, -52.3223953, 24.6415558, -76.9447861, 76.9083939
33: -72.2059021, 33.7055702, -72.2850418, 33.7871399, -105.6650696, 105.7388458
34: -65.4445496, 17.0680542, -65.4691849, 17.1028671, -81.6520691, 81.7467880
35: -63.7068977, 23.5035191, -63.7165260, 23.5349102, -85.7086639, 85.7476196
36: -61.9178276, 24.3516369, -61.9244537, 24.3648834, -86.2827148, 86.2760925
37: -87.0681686, 19.7341480, -87.1170578, 19.7634201, -106.8315887, 106.8512039
38: -69.8601990, 29.1393490, -69.9141083, 29.1331501, -98.9933472, 99.0534592
39: -80.3273392, 30.5441360, -80.3684311, 30.5480633, -110.8754044, 110.9125671
40: -62.4911461, 25.5654812, -62.5165749, 25.5243683, -88.0155182, 88.0820541
41: -54.9321480, 32.7677841, -54.9503365, 32.7855110, -87.7176590, 87.7181244
42: -36.1867867, 25.8846970, -36.1866646, 25.9664612, -62.1532478, 62.0713615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1379558
time: 109.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
time: 75.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -56.5152054, 43.4793015, -56.6614876, 43.5468674, -100.0620728, 100.1407928
1: -25.2557392, 37.7535019, -25.3643990, 37.8360405, -63.0917816, 63.1179008
2: -21.7506828, 37.1640968, -21.9093266, 37.2741356, -59.0248184, 59.0734253
3: -24.3623466, 39.7848053, -24.5461826, 39.9162445, -64.2785950, 64.3309860
4: -28.4212093, 43.7290573, -28.5958805, 43.8377037, -72.2589111, 72.3249359
5: -24.5780678, 39.7384415, -24.7466469, 39.8496552, -64.4277191, 64.4850922
6: -54.2309647, 31.7680531, -54.3012772, 31.8707581, -86.1017227, 86.0693283
7: -30.3765602, 39.5853500, -30.5184593, 39.6412125, -70.0177765, 70.1038055
8: -36.5276260, 53.5621834, -36.6695328, 53.6815643, -90.2091904, 90.2317200
9: -29.0537128, 39.0067825, -29.1332588, 39.0570984, -68.1108093, 68.1400452
10: -49.5323868, 43.7234688, -49.6402206, 43.9037552, -93.4361420, 93.3636932
11: -48.9885368, 21.8262196, -49.2023048, 22.0390778, -71.0276184, 71.0285263
12: -55.2319336, 25.0472164, -55.3659134, 25.2791595, -79.4032364, 79.2507324
13: -50.4974861, 43.7205658, -50.6425629, 43.8368416, -94.3343277, 94.3631287
14: -87.4380875, 31.0199299, -87.6165466, 31.2671242, -118.7052155, 118.6364746
15: -35.6739006, 36.0056992, -35.8391342, 36.0716591, -71.7455597, 71.8448334
16: -45.8440666, 33.7814407, -45.9619904, 33.9203873, -79.7644501, 79.7434311
17: -84.7938309, 23.3464165, -84.9816284, 23.5399055, -108.3337402, 108.3280487
18: -48.9197159, 31.2439938, -49.0965652, 31.3547363, -80.2744522, 80.3405609
19: -38.8670120, 18.4445267, -39.0198746, 18.5858955, -57.4529076, 57.4644012
20: -36.8626785, 23.2844906, -36.9863968, 23.4136696, -60.2763481, 60.2708893
21: -48.0579224, 21.9791241, -48.2447701, 22.1546326, -70.2125549, 70.2238922
22: -49.9733963, 22.0426521, -50.0686798, 22.1521759, -72.1255722, 72.1113281
23: -39.0461273, 23.9270668, -39.1722412, 24.0724525, -63.1185799, 63.0993080
24: -46.2784576, 23.9521770, -46.3832588, 24.0568466, -70.3353043, 70.3354340
25: -41.2108231, 24.7152996, -41.2979126, 24.8346348, -66.0454559, 66.0132141
26: -56.8919411, 33.3725128, -57.0691643, 33.6260567, -90.5179977, 90.4416809
27: -45.2193756, 28.6707687, -45.3582306, 28.7550278, -73.9744034, 74.0289993
28: -38.8833351, 26.6311035, -39.0093803, 26.7766476, -65.6599808, 65.6404877
29: -51.6171150, 20.5192680, -51.7328987, 20.6574039, -72.2745209, 72.2521667
30: -49.2336121, 26.0672855, -49.3742142, 26.2477016, -75.4813156, 75.4414978
31: -50.9748306, 27.7707767, -51.1510429, 27.9263020, -78.9011307, 78.9218216
32: -52.3256302, 24.5933571, -52.4044876, 24.6827354, -77.0083618, 76.9978485
33: -72.2201309, 33.7181396, -72.3600693, 33.8412704, -105.7366028, 105.9467697
34: -65.4586639, 17.0799026, -65.5256805, 17.1640358, -81.7195435, 81.8941040
35: -63.7307358, 23.5137653, -63.7960434, 23.5722790, -85.7497406, 85.8444824
36: -61.9389229, 24.3602142, -61.9995651, 24.4083195, -86.3472443, 86.3597794
37: -87.0865021, 19.7530880, -87.1739197, 19.8313389, -106.9178391, 106.9270096
38: -69.8759384, 29.1691914, -69.9744263, 29.2233887, -99.0993271, 99.1436157
39: -80.3442993, 30.5809555, -80.4632721, 30.6617622, -111.0060577, 111.0442276
40: -62.5046463, 25.6284370, -62.5873375, 25.7002487, -88.2048950, 88.2157745
41: -54.9422684, 32.7941933, -55.0112762, 32.8775063, -87.8197784, 87.8054657
42: -36.1983795, 25.8981628, -36.2337570, 26.0228348, -62.2212143, 62.1319199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1217664
time: 70.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2352701
time: 80.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -56.6886597, 43.5076714, -56.5862389, 43.4063950, -100.0950546, 100.0939102
1: -25.3887749, 37.7968826, -25.3115578, 37.6997833, -63.0885582, 63.1084404
2: -21.9542313, 37.2474403, -21.8759785, 37.1752586, -59.1294899, 59.1234207
3: -24.5989227, 39.9018021, -24.5612469, 39.8496857, -64.4486084, 64.4630508
4: -28.6433735, 43.7861290, -28.5301819, 43.6533585, -72.2967300, 72.3163147
5: -24.7937889, 39.8213768, -24.7304668, 39.7372513, -64.5310364, 64.5518417
6: -54.3062286, 31.8364906, -54.2768631, 31.8037872, -86.1100159, 86.1133575
7: -30.5503979, 39.5911407, -30.4509602, 39.4745445, -70.0249405, 70.0420990
8: -36.7066154, 53.6391602, -36.6194077, 53.5288124, -90.2354279, 90.2585678
9: -29.1467190, 39.0581818, -29.1116161, 38.9964828, -68.1432037, 68.1697998
10: -49.6448746, 43.9279175, -49.5852928, 43.8790016, -93.5238800, 93.5132141
11: -49.1699791, 22.1044464, -49.0568008, 22.0392227, -71.2091980, 71.1612473
12: -55.3189545, 25.3462677, -55.1933594, 25.2396431, -79.4463043, 79.3784637
13: -50.6272964, 43.8464432, -50.6054459, 43.8124771, -94.4397736, 94.4518890
14: -87.5787354, 31.3430119, -87.4349060, 31.2390060, -118.8177414, 118.7779160
15: -35.8167572, 36.0755157, -35.7813263, 36.0511665, -71.8679199, 71.8568420
16: -45.9710617, 33.8618774, -45.9182205, 33.7996750, -79.7707367, 79.7800980
17: -84.9186325, 23.5947819, -84.7433853, 23.5053310, -108.4239655, 108.3381653
18: -49.0935440, 31.4492111, -49.0998230, 31.3963223, -80.4898682, 80.5490341
19: -39.0060577, 18.6271725, -38.9349823, 18.5960045, -57.6020622, 57.5621567
20: -36.9769478, 23.4507847, -36.9093018, 23.4063187, -60.3832664, 60.3600845
21: -48.2157173, 22.2063942, -48.1073685, 22.1468315, -70.3625488, 70.3137665
22: -50.0012283, 22.1819458, -49.8330841, 22.0955353, -72.0967636, 72.0150299
23: -39.1440659, 24.1128597, -39.0495110, 24.0487728, -63.1928406, 63.1623688
24: -46.3616180, 24.0829582, -46.2821503, 24.0462265, -70.4078445, 70.3651123
25: -41.2602768, 24.8649635, -41.1510849, 24.7822361, -66.0425110, 66.0160522
26: -57.0314331, 33.7045212, -56.9075012, 33.6166611, -90.6480942, 90.6120224
27: -45.3604774, 28.7765617, -45.3262978, 28.7567959, -74.1172714, 74.1028595
28: -38.9937363, 26.8209648, -38.9111786, 26.7577705, -65.7515106, 65.7321472
29: -51.6540413, 20.7000237, -51.4520111, 20.5942383, -72.2482758, 72.1520386
30: -49.3375778, 26.2963066, -49.2270584, 26.2162819, -75.5538635, 75.5233612
31: -51.1413040, 27.9681854, -51.0716324, 27.9320488, -79.0733490, 79.0398178
32: -52.3981590, 24.6888943, -52.3430939, 24.6715527, -77.0697098, 77.0319901
33: -72.3997345, 33.8479118, -72.3471451, 33.8091621, -105.8830109, 105.9524002
34: -65.5286560, 17.1697636, -65.4917679, 17.1270161, -81.7821350, 81.8792114
35: -63.7758331, 23.5739594, -63.7358627, 23.5491238, -85.8140488, 85.8561096
36: -61.9830933, 24.4186077, -61.9399643, 24.3812809, -86.3643723, 86.3585739
37: -87.1711578, 19.8465195, -87.1383972, 19.7984657, -106.9696198, 106.9849167
38: -69.9742355, 29.2099133, -69.9431992, 29.1477757, -99.1220093, 99.1531143
39: -80.4806824, 30.6438351, -80.4147720, 30.5630379, -111.0437164, 111.0586090
40: -62.5909081, 25.6408501, -62.5389938, 25.5395145, -88.1304245, 88.1798401
41: -55.0162659, 32.8616028, -54.9717827, 32.8109703, -87.8272400, 87.8333893
42: -36.2552834, 26.0058556, -36.2024689, 25.9985332, -62.2538147, 62.2083244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1431511
time: 91.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
time: 122.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -56.7059441, 43.5586662, -56.7168846, 43.5607262, -100.2666702, 100.2755508
1: -25.3984089, 37.8467445, -25.4074306, 37.8450089, -63.2434158, 63.2541733
2: -21.9633923, 37.2846527, -21.9779549, 37.2842064, -59.2475967, 59.2626076
3: -24.6053028, 39.9268036, -24.6248302, 39.9284706, -64.5337753, 64.5516357
4: -28.6545734, 43.8529472, -28.6706829, 43.8481522, -72.5027237, 72.5236282
5: -24.8022881, 39.8623657, -24.8180561, 39.8617783, -64.6640625, 64.6804199
6: -54.3157349, 31.8643036, -54.3196487, 31.8970757, -86.2128143, 86.1839523
7: -30.5632229, 39.6512222, -30.5738659, 39.6497574, -70.2129822, 70.2250900
8: -36.7154999, 53.6963539, -36.7283401, 53.6959229, -90.4114227, 90.4246979
9: -29.1546669, 39.0814590, -29.1630554, 39.0751190, -68.2297821, 68.2445145
10: -49.6675072, 43.9439240, -49.6660538, 43.9694138, -93.6369171, 93.6099777
11: -49.2240982, 22.1135521, -49.2236404, 22.1351070, -71.3592072, 71.3371887
12: -55.3816147, 25.3565292, -55.3804054, 25.3789787, -79.6560745, 79.5679321
13: -50.6437836, 43.8600388, -50.6859741, 43.8631706, -94.5069580, 94.5460129
14: -87.6446609, 31.3506966, -87.6465988, 31.3778782, -119.0225372, 118.9972992
15: -35.8399277, 36.0851288, -35.8844032, 36.0889664, -71.9288940, 71.9695282
16: -45.9887009, 33.9123840, -45.9966469, 33.9588318, -79.9475327, 79.9090271
17: -85.0026474, 23.6052418, -85.0007553, 23.6248398, -108.6274872, 108.6059952
18: -49.1098404, 31.4486160, -49.1128006, 31.4196320, -80.5294724, 80.5614166
19: -39.0370865, 18.6315651, -39.0399170, 18.6478901, -57.6849747, 57.6714821
20: -37.0044556, 23.4560013, -37.0079651, 23.4704304, -60.4748840, 60.4639664
21: -48.2650261, 22.2127781, -48.2675476, 22.2328186, -70.4978485, 70.4803238
22: -50.0871735, 22.1887035, -50.0869370, 22.1974869, -72.2846603, 72.2756424
23: -39.1888161, 24.1186943, -39.1895714, 24.1349907, -63.3238068, 63.3082657
24: -46.3983307, 24.0879974, -46.4014320, 24.1015148, -70.4998474, 70.4894257
25: -41.3137207, 24.8738976, -41.3149109, 24.8862991, -66.2000198, 66.1888123
26: -57.0877724, 33.7120857, -57.0909500, 33.7373657, -90.8251343, 90.8030396
27: -45.3760414, 28.7807999, -45.3808937, 28.7859516, -74.1619949, 74.1616974
28: -39.0291100, 26.8262577, -39.0295639, 26.8406792, -65.8697891, 65.8558197
29: -51.7527733, 20.7058792, -51.7494469, 20.7185936, -72.4713669, 72.4553223
30: -49.3920784, 26.3059692, -49.3926010, 26.3251381, -75.7172165, 75.6985703
31: -51.1720848, 27.9745502, -51.1750298, 27.9941025, -79.1661835, 79.1495819
32: -52.4211311, 24.6959496, -52.4248581, 24.7119446, -77.1330719, 77.1208038
33: -72.4139099, 33.8605042, -72.4219055, 33.8630829, -105.9545746, 106.1624603
34: -65.5436401, 17.1817226, -65.5476532, 17.1885586, -81.8523712, 82.0255966
35: -63.8000679, 23.5845261, -63.8150864, 23.5863380, -85.8542099, 85.9576721
36: -62.0045509, 24.4273109, -62.0145531, 24.4245949, -86.4291458, 86.4418640
37: -87.1898041, 19.8650818, -87.1948547, 19.8648720, -107.0546722, 107.0599365
38: -69.9902802, 29.2405720, -70.0035019, 29.2382755, -99.2285538, 99.2440720
39: -80.4976578, 30.6807308, -80.5092163, 30.6764965, -111.1741562, 111.1899490
40: -62.6053314, 25.6994019, -62.6097298, 25.7142067, -88.3195343, 88.3091278
41: -55.0268097, 32.8896408, -55.0325775, 32.9025803, -87.9293900, 87.9222183
42: -36.2684479, 26.0203819, -36.2493172, 26.0552559, -62.3237038, 62.2696991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 932

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1312242
time: 105.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2379575
time: 88.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 196.55 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1868988
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2301780
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1204245
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2338611
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1928414
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1306175
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2363706
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1346312
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2302435
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1204421
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2340991
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1405458
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1307067
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1307067
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1898741
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1217664
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2352701
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -44.0636384, upper bound: 44.1950421
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -44.0636384, upper bound: 44.2366513
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1312242
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2379575
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1379558
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1217664
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2352701
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1431511
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.1312242
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 196.55
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2379575

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -56.2838898, 43.3628235, -56.3226280, 43.2826805, -99.5665741, 99.6854553
1: -25.0701771, 37.5905190, -25.1135349, 37.5795288, -62.6497040, 62.7040558
2: -21.5192909, 37.0062752, -21.6180706, 37.0800362, -58.5993271, 58.6243439
3: -24.0823097, 39.5802689, -24.3399906, 39.7607040, -63.8430138, 63.9202576
4: -28.1401024, 43.5171661, -28.1937332, 43.4863625, -71.6264648, 71.7108994
5: -24.3297215, 39.5641594, -24.4818420, 39.5980034, -63.9277267, 64.0460052
6: -54.1300583, 31.6586723, -54.1695633, 31.6729889, -85.8030472, 85.8282318
7: -30.1581707, 39.4502106, -30.1903725, 39.3180313, -69.4762039, 69.6405792
8: -36.3099136, 53.3380089, -36.3849640, 53.3823318, -89.6922455, 89.7229767
9: -28.9368210, 38.8790474, -28.9970589, 38.8957596, -67.8325806, 67.8761063
10: -49.2243881, 43.2906189, -49.3908081, 43.5100403, -92.7344284, 92.6814270
11: -48.6220932, 21.4458542, -48.8623352, 21.6884766, -70.3105698, 70.3081894
12: -55.0003815, 24.6883469, -54.9914284, 24.8543568, -78.7325058, 78.5562897
13: -50.2389221, 43.4989891, -50.4127655, 43.6854973, -93.9244232, 93.9117584
14: -87.0734711, 30.5592594, -87.0866699, 30.7086830, -117.7821503, 117.6459274
15: -35.4462357, 35.8944855, -35.5727654, 35.9632912, -71.4095306, 71.4672546
16: -45.6141739, 33.5250969, -45.7288780, 33.5846519, -79.1988220, 79.2539749
17: -84.4534149, 23.0290070, -84.4471741, 23.1410351, -107.5944519, 107.4761810
18: -48.6522827, 30.9375114, -49.0214577, 31.2297459, -79.8820267, 79.9589691
19: -38.6147766, 18.2189522, -38.8142014, 18.4299126, -57.0446892, 57.0331535
20: -36.6631775, 23.0726471, -36.7800980, 23.1993828, -59.8625603, 59.8527451
21: -47.7500725, 21.6915512, -47.9456444, 21.8975182, -69.6475906, 69.6371918
22: -49.7657356, 21.9007339, -49.5978813, 21.8858109, -71.6515503, 71.4986115
23: -38.8096237, 23.6726761, -38.8864822, 23.7975082, -62.6071320, 62.5591583
24: -46.0623856, 23.7674122, -46.1220474, 23.8615189, -69.9239044, 69.8894577
25: -40.9971581, 24.5050964, -40.9629822, 24.5181866, -65.5153427, 65.4680786
26: -56.6265640, 32.9844131, -56.7005005, 33.2473297, -89.8738937, 89.6849136
27: -45.0709496, 28.5651836, -45.2047958, 28.6459618, -73.7169113, 73.7699814
28: -38.6991806, 26.4359016, -38.7689209, 26.5342751, -65.2334595, 65.2048187
29: -51.3703117, 20.3320065, -51.1944008, 20.3163471, -71.6866608, 71.5264053
30: -48.9807434, 25.7594280, -49.0538254, 25.8925362, -74.8732758, 74.8132553
31: -50.6593094, 27.4823799, -50.9245453, 27.7031860, -78.3624954, 78.4069214
32: -52.2000504, 24.5018272, -52.2254639, 24.5747700, -76.7748184, 76.7272949
33: -71.9454498, 33.5255165, -72.0824814, 33.6712723, -105.3004913, 105.2659912
34: -65.3098907, 16.9518509, -65.3766785, 17.0161610, -81.4614182, 81.4391098
35: -63.5556831, 23.3663883, -63.6092873, 23.4804459, -85.4715271, 85.3849640
36: -61.7658997, 24.2313328, -61.8086090, 24.3058910, -86.0717926, 86.0399399
37: -86.9424896, 19.6033802, -86.9476166, 19.6281128, -106.5706024, 106.5509949
38: -69.6515198, 28.9810791, -69.7496796, 29.0016632, -98.6531830, 98.7307587
39: -80.1085587, 30.4092941, -80.1509552, 30.3991127, -110.5076752, 110.5602493
40: -62.3639450, 25.4895744, -62.3663216, 25.3546410, -87.7185822, 87.8558960
41: -54.8002167, 32.6705513, -54.7964783, 32.6609039, -87.4611206, 87.4670258
42: -36.0994034, 25.7552929, -36.1457443, 25.8790073, -61.9784088, 61.9010391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1826639
time: 86.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2301780
time: 73.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -56.2895699, 43.3888855, -56.4507599, 43.4324799, -99.7220459, 99.8396454
1: -25.0735149, 37.6114807, -25.2083492, 37.7243462, -62.7978592, 62.8198318
2: -21.5214958, 37.0224304, -21.7184258, 37.1888199, -58.7103157, 58.7408562
3: -24.0859318, 39.5937653, -24.4023933, 39.8388062, -63.9247360, 63.9961586
4: -28.1410770, 43.5359573, -28.3320637, 43.6804276, -71.8215027, 71.8680191
5: -24.3329601, 39.5773468, -24.5683403, 39.7220306, -64.0549927, 64.1456909
6: -54.1334877, 31.6653938, -54.2099915, 31.7618256, -85.8953094, 85.8753815
7: -30.1621761, 39.4730682, -30.3119221, 39.4925613, -69.6547394, 69.7849884
8: -36.3134232, 53.3642654, -36.4923935, 53.5485535, -89.8619766, 89.8566589
9: -28.9384308, 38.8976669, -29.0458794, 38.9734039, -67.9118347, 67.9435425
10: -49.2105637, 43.2991753, -49.4707336, 43.5979042, -92.8084717, 92.7699127
11: -48.6363220, 21.4496174, -49.0287704, 21.7821026, -70.4184265, 70.4783859
12: -55.0172882, 24.6950359, -55.1774445, 24.9924088, -78.8822479, 78.7377625
13: -50.2471924, 43.5045471, -50.4972229, 43.7342377, -93.9814301, 94.0017700
14: -87.0661774, 30.5636520, -87.2967834, 30.8455429, -117.9117203, 117.8604355
15: -35.4540710, 35.8975143, -35.6727066, 35.9991264, -71.4532013, 71.5702209
16: -45.6208344, 33.5463600, -45.8059311, 33.7374344, -79.3582687, 79.3522949
17: -84.4693298, 23.0339527, -84.7038574, 23.2578831, -107.7272110, 107.7378082
18: -48.6536980, 30.9565029, -49.0331039, 31.2641754, -79.9178772, 79.9896088
19: -38.6320419, 18.2217274, -38.9183464, 18.4808693, -57.1129112, 57.1400757
20: -36.6740189, 23.0740356, -36.8780060, 23.2629566, -59.9369736, 59.9520416
21: -47.7744408, 21.6939774, -48.1044655, 21.9824677, -69.7569122, 69.7984467
22: -49.7991791, 21.9023762, -49.8497620, 21.9867020, -71.7858810, 71.7521362
23: -38.8232117, 23.6743984, -39.0262260, 23.8825970, -62.7058105, 62.7006226
24: -46.0739174, 23.7689133, -46.2408371, 23.9158459, -69.9897614, 70.0097504
25: -41.0123215, 24.5037327, -41.1267433, 24.6200657, -65.6323853, 65.6304779
26: -56.6467094, 32.9900742, -56.8831444, 33.3667603, -90.0134735, 89.8732147
27: -45.0777664, 28.5694294, -45.2577133, 28.6837673, -73.7615356, 73.8271408
28: -38.7143745, 26.4370918, -38.8870087, 26.6160870, -65.3304596, 65.3240967
29: -51.4088058, 20.3329144, -51.4892540, 20.4394646, -71.8482666, 71.8221664
30: -48.9978676, 25.7619743, -49.2188225, 25.9997387, -74.9976044, 74.9807968
31: -50.6717072, 27.4852161, -51.0272102, 27.7637749, -78.4354858, 78.5124283
32: -52.2103004, 24.5049191, -52.3020630, 24.6150150, -76.8253174, 76.8069839
33: -71.9440308, 33.5160675, -72.1550369, 33.7246552, -105.4668961, 105.4309082
34: -65.3218002, 16.9559669, -65.4301758, 17.0744801, -81.6128693, 81.5569153
35: -63.5646057, 23.3716011, -63.6862144, 23.5171909, -85.5382614, 85.4572449
36: -61.7766190, 24.2434006, -61.8822937, 24.3487492, -86.1253662, 86.1256943
37: -86.9473877, 19.6062431, -86.9970703, 19.6897888, -106.6371765, 106.6033173
38: -69.6576309, 28.9732780, -69.8052292, 29.0832653, -98.7408981, 98.7785034
39: -80.1148376, 30.4091949, -80.2424545, 30.5121021, -110.6269379, 110.6516495
40: -62.3689461, 25.4925079, -62.4239845, 25.5136337, -87.8825836, 87.9164886
41: -54.8033867, 32.6663055, -54.8516846, 32.7436218, -87.5470123, 87.5179901
42: -36.1183319, 25.7576866, -36.1917114, 25.9300232, -62.0483551, 61.9493980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1861226
time: 91.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2338611
time: 93.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -56.4724731, 43.4402275, -56.3778954, 43.2969666, -99.7694397, 99.8181229
1: -25.2109451, 37.6824112, -25.1568127, 37.5886917, -62.7996368, 62.8392258
2: -21.7298260, 37.1253853, -21.6870155, 37.0903702, -58.8201981, 58.8124008
3: -24.3229465, 39.7200546, -24.4190750, 39.7731590, -64.0961075, 64.1391296
4: -28.3703060, 43.6369324, -28.2689552, 43.4968910, -71.8671951, 71.9058838
5: -24.5522213, 39.6865692, -24.5537510, 39.6103630, -64.1625824, 64.2403183
6: -54.2116013, 31.7495289, -54.1880798, 31.6999664, -85.9115677, 85.9376068
7: -30.3420162, 39.5162506, -30.2461281, 39.3271103, -69.6691284, 69.7623749
8: -36.4947624, 53.4705963, -36.4441681, 53.3970032, -89.8917694, 89.9147644
9: -29.0327530, 38.9504051, -29.0260353, 38.9147339, -67.9474869, 67.9764404
10: -49.3544235, 43.5074463, -49.4167442, 43.5765076, -92.9309311, 92.9241943
11: -48.8525238, 21.7301903, -48.8842010, 21.7845268, -70.6370544, 70.6143951
12: -55.1498222, 24.9953747, -55.0062027, 24.9551201, -78.9846191, 78.8574524
13: -50.3829460, 43.6381416, -50.4625931, 43.7121010, -94.0950470, 94.1007385
14: -87.2752533, 30.8876019, -87.1170731, 30.8200397, -118.0952911, 118.0046768
15: -35.5937309, 35.9715805, -35.6209412, 35.9807053, -71.5744324, 71.5925217
16: -45.7514191, 33.6518974, -45.7637329, 33.6217194, -79.3731384, 79.4156342
17: -84.6611099, 23.2837696, -84.4668274, 23.2261734, -107.8872833, 107.7505951
18: -48.8398438, 31.1375065, -49.0383148, 31.3066769, -80.1465225, 80.1758194
19: -38.7813225, 18.4041901, -38.8346481, 18.4921188, -57.2734413, 57.2388382
20: -36.8032608, 23.2431984, -36.8019524, 23.2563019, -60.0595627, 60.0451508
21: -47.9539757, 21.9235420, -47.9686852, 21.9758091, -69.9297867, 69.8922272
22: -49.8778114, 22.0424271, -49.6173592, 21.9311943, -71.8090057, 71.6597900
23: -38.9499168, 23.8620796, -38.9040794, 23.8601456, -62.8100624, 62.7661591
24: -46.1811371, 23.9021816, -46.1411819, 23.9059658, -70.0871048, 70.0433655
25: -41.0983772, 24.6615543, -40.9807663, 24.5697327, -65.6681061, 65.6423187
26: -56.8209343, 33.3184738, -56.7229996, 33.3586998, -90.1796341, 90.0414734
27: -45.2235832, 28.6738529, -45.2281227, 28.6939850, -73.9175720, 73.9019775
28: -38.8435936, 26.6289806, -38.7894363, 26.5984230, -65.4420166, 65.4184189
29: -51.5050888, 20.5164642, -51.2112236, 20.3777218, -71.8828125, 71.7276917
30: -49.1356125, 25.9955826, -49.0727005, 25.9700127, -75.1056213, 75.0682831
31: -50.8530159, 27.6843739, -50.9489288, 27.7706680, -78.6236877, 78.6333008
32: -52.2956429, 24.6019058, -52.2466354, 24.6048126, -76.9004517, 76.8485413
33: -72.1375427, 33.6654015, -72.1442795, 33.6934395, -105.4972534, 105.4765625
34: -65.3914337, 17.0484104, -65.3997650, 17.0404968, -81.5747375, 81.5720367
35: -63.6227074, 23.4346752, -63.6285400, 23.4946671, -85.5671768, 85.4959564
36: -61.8286362, 24.2970276, -61.8236923, 24.3223133, -86.1509476, 86.1207199
37: -87.0415649, 19.7179642, -86.9681244, 19.6651058, -106.7066727, 106.6860886
38: -69.7609711, 29.0484657, -69.7787476, 29.0169888, -98.7779617, 98.8272095
39: -80.2559052, 30.5072575, -80.1966095, 30.4142628, -110.6701660, 110.7038651
40: -62.4609985, 25.5636024, -62.3884811, 25.3685818, -87.8295822, 87.9520874
41: -54.8810577, 32.7637558, -54.8176346, 32.6855125, -87.5665741, 87.5813904
42: -36.1653900, 25.8698559, -36.1615372, 25.9103432, -62.0757332, 62.0313950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1880775
time: 84.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
time: 83.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -56.4782295, 43.4682770, -56.5061493, 43.4465027, -99.9247284, 99.9744263
1: -25.2141895, 37.7030449, -25.2515240, 37.7333031, -62.9474945, 62.9545670
2: -21.7321129, 37.1412773, -21.7871094, 37.1988907, -58.9310036, 58.9283867
3: -24.3264885, 39.7331352, -24.4810333, 39.8509674, -64.1774597, 64.2141724
4: -28.3710518, 43.6585617, -28.4068432, 43.6909294, -72.0619812, 72.0654068
5: -24.5552750, 39.6997452, -24.6398506, 39.7341766, -64.2894516, 64.3395996
6: -54.2151031, 31.7549725, -54.2283707, 31.7884216, -86.0035248, 85.9833450
7: -30.3457012, 39.5378952, -30.3673496, 39.5011444, -69.8468475, 69.9052429
8: -36.4984894, 53.4963837, -36.5513268, 53.5629578, -90.0614471, 90.0477142
9: -29.0346680, 38.9694595, -29.0756168, 38.9918938, -68.0265656, 68.0450745
10: -49.3408737, 43.5160370, -49.4966049, 43.6637115, -93.0045853, 93.0126419
11: -48.8666077, 21.7334595, -49.0500565, 21.8779049, -70.7445145, 70.7835159
12: -55.1661911, 25.0020370, -55.1920471, 25.0923309, -79.1346130, 79.0395279
13: -50.3842545, 43.6435471, -50.5405922, 43.7608719, -94.1451263, 94.1841431
14: -87.2697754, 30.8920803, -87.3271027, 30.9561443, -118.2259216, 118.2191849
15: -35.5976372, 35.9745636, -35.7190857, 36.0163307, -71.6139679, 71.6936493
16: -45.7581711, 33.6743088, -45.8405190, 33.7764702, -79.5346375, 79.5148315
17: -84.6763153, 23.2887554, -84.7230911, 23.3426170, -108.0189362, 108.0118484
18: -48.8403664, 31.1455746, -49.0494080, 31.3288517, -80.1692200, 80.1949844
19: -38.7982941, 18.4072227, -38.9383812, 18.5429211, -57.3412170, 57.3456039
20: -36.8140984, 23.2444668, -36.8996773, 23.3196468, -60.1337433, 60.1441422
21: -47.9782791, 21.9263229, -48.1272888, 22.0606613, -70.0389404, 70.0536118
22: -49.9118004, 22.0440025, -49.8683586, 22.0318470, -71.9436493, 71.9123611
23: -38.9638443, 23.8638458, -39.0435829, 23.9450665, -62.9089127, 62.9074287
24: -46.1922722, 23.9036427, -46.2592430, 23.9604225, -70.1526947, 70.1628876
25: -41.1125717, 24.6598186, -41.1437225, 24.6716633, -65.7842331, 65.8035431
26: -56.8397484, 33.3244591, -56.9051437, 33.4777985, -90.3175507, 90.2295990
27: -45.2305527, 28.6674881, -45.2807274, 28.7164879, -73.9470367, 73.9482117
28: -38.8588028, 26.6300888, -38.9072723, 26.6801090, -65.5389099, 65.5373611
29: -51.5432854, 20.5173473, -51.5059090, 20.5006123, -72.0438995, 72.0232544
30: -49.1525993, 25.9972725, -49.2371750, 26.0769157, -75.2295151, 75.2344513
31: -50.8652229, 27.6873779, -51.0512085, 27.8315392, -78.6967621, 78.7385864
32: -52.3046875, 24.6044922, -52.3230095, 24.6443539, -76.9490433, 76.9275055
33: -72.1356583, 33.6557159, -72.2166443, 33.7466660, -105.6629028, 105.6434631
34: -65.4038696, 17.0525017, -65.4524994, 17.0992374, -81.7261887, 81.6853409
35: -63.6331482, 23.4400883, -63.7051888, 23.5312767, -85.6374130, 85.5723114
36: -61.8389282, 24.3091850, -61.8968239, 24.3651218, -86.2040482, 86.2060089
37: -87.0463409, 19.7144966, -87.0172882, 19.7241154, -106.7704544, 106.7317810
38: -69.7663727, 29.0430069, -69.8339081, 29.0985203, -98.8648911, 98.8769150
39: -80.2618561, 30.5065956, -80.2878647, 30.5270557, -110.7889099, 110.7944641
40: -62.4662514, 25.5574112, -62.4461098, 25.5266418, -87.9928894, 88.0035248
41: -54.8844910, 32.7569809, -54.8728142, 32.7677002, -87.6521912, 87.6297913
42: -36.1868439, 25.8712692, -36.2072983, 25.9618988, -62.1487427, 62.0785675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1888820
time: 91.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2363706
time: 92.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -56.2838898, 43.3628235, -56.4733620, 43.3767090, -99.6605988, 99.8361816
1: -25.0701771, 37.5905190, -25.2165279, 37.6781082, -62.7482834, 62.8070450
2: -21.5192909, 37.0062752, -21.7372360, 37.1514282, -58.6707191, 58.7435112
3: -24.0823097, 39.5802689, -24.3930779, 39.8181763, -63.9004860, 63.9733467
4: -28.1401024, 43.5171661, -28.3706722, 43.6292114, -71.7693176, 71.8878403
5: -24.3297215, 39.5641594, -24.5834885, 39.7071304, -64.0368500, 64.1476440
6: -54.1300583, 31.6586723, -54.2331009, 31.7559261, -85.8859863, 85.8917694
7: -30.1581707, 39.4502106, -30.3358154, 39.4533615, -69.6115341, 69.7860260
8: -36.3099136, 53.3380089, -36.4956207, 53.4930878, -89.8030014, 89.8336334
9: -28.9368210, 38.8790474, -29.0517673, 38.9532166, -67.8900375, 67.9308167
10: -49.2243881, 43.2906189, -49.5241928, 43.6793594, -92.9037476, 92.8148117
11: -48.6220932, 21.4458542, -49.0065079, 21.8171463, -70.4392395, 70.4523621
12: -55.0003815, 24.6883469, -55.1590958, 25.0275230, -78.8952637, 78.7094269
13: -50.2389221, 43.4989891, -50.4807205, 43.7547073, -93.9936295, 93.9797058
14: -87.0734711, 30.5592594, -87.3650131, 30.9750137, -118.0484848, 117.9242706
15: -35.4462357, 35.8944855, -35.6741028, 36.0096436, -71.4558792, 71.5685883
16: -45.6141739, 33.5250969, -45.8354378, 33.7084045, -79.3225784, 79.3605347
17: -84.4534149, 23.0290070, -84.7003937, 23.3197746, -107.7731934, 107.7294006
18: -48.6522827, 30.9375114, -49.0595245, 31.2270279, -79.8793106, 79.9970398
19: -38.6147766, 18.2189522, -38.8864670, 18.4604836, -57.0752602, 57.1054192
20: -36.6631775, 23.0726471, -36.8605118, 23.2817116, -59.9448891, 59.9331589
21: -47.7500725, 21.6915512, -48.0549431, 21.9753933, -69.7254639, 69.7464905
22: -49.7657356, 21.9007339, -49.7903671, 22.0080643, -71.7738037, 71.6911011
23: -38.8096237, 23.6726761, -39.0081444, 23.9053993, -62.7150230, 62.6808205
24: -46.0623856, 23.7674122, -46.2370758, 23.9416676, -70.0040512, 70.0044861
25: -40.9971581, 24.5050964, -41.1085281, 24.6650276, -65.6621857, 65.6136246
26: -56.6265640, 32.9844131, -56.8578491, 33.3820724, -90.0086365, 89.8422623
27: -45.0709496, 28.5651836, -45.2730370, 28.6846008, -73.7555542, 73.8382187
28: -38.6991806, 26.4359016, -38.8647308, 26.6328850, -65.3320618, 65.3006287
29: -51.3703117, 20.3320065, -51.4150238, 20.4745293, -71.8448410, 71.7470322
30: -48.9807434, 25.7594280, -49.1831779, 26.0433140, -75.0240555, 74.9426041
31: -50.6593094, 27.4823799, -51.0124855, 27.7696209, -78.4289322, 78.4948654
32: -52.2000504, 24.5018272, -52.2979355, 24.6190033, -76.8190536, 76.7997589
33: -71.9454498, 33.5255165, -72.2035675, 33.7651634, -105.4753265, 105.5002441
34: -65.3098907, 16.9518509, -65.4331818, 17.0756245, -81.5711670, 81.5861893
35: -63.5556831, 23.3663883, -63.6709976, 23.5167065, -85.5747528, 85.5376129
36: -61.7658997, 24.2313328, -61.8772583, 24.3479252, -86.1138229, 86.1085892
37: -86.9424896, 19.6033802, -87.0845032, 19.7340374, -106.6765289, 106.6878815
38: -69.6515198, 28.9810791, -69.8537903, 29.1106586, -98.7621765, 98.8348694
39: -80.1085587, 30.4092941, -80.3012466, 30.5312176, -110.6397781, 110.7105408
40: -62.3639450, 25.4895744, -62.4832077, 25.5161533, -87.8800964, 87.9727783
41: -54.8002167, 32.6705513, -54.9129372, 32.7646332, -87.5648499, 87.5834885
42: -36.0994034, 25.7552929, -36.1679459, 25.9330978, -62.0325012, 61.9232407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1829501
time: 107.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2302435
time: 84.29 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -56.3026314, 43.4142113, -56.6037140, 43.5313759, -99.8340073, 100.0179291
1: -25.0809708, 37.6407280, -25.3123779, 37.8234673, -62.9044380, 62.9531059
2: -21.5298538, 37.0436554, -21.8392544, 37.2606583, -58.7905121, 58.8829117
3: -24.0898075, 39.6054230, -24.4568634, 39.8972931, -63.9870987, 64.0622864
4: -28.1524315, 43.5820465, -28.5112991, 43.8240967, -71.9765320, 72.0933456
5: -24.3393326, 39.6056671, -24.6713181, 39.8319206, -64.1712494, 64.2769852
6: -54.1398735, 31.6841583, -54.2759705, 31.8506107, -85.9904861, 85.9601288
7: -30.1725998, 39.5117035, -30.4588890, 39.6289711, -69.8015747, 69.9705963
8: -36.3199234, 53.3949661, -36.6045341, 53.6605148, -89.9804382, 89.9994965
9: -28.9450359, 38.9071808, -29.1023846, 39.0331650, -67.9782028, 68.0095673
10: -49.2454834, 43.3087959, -49.6050034, 43.7703781, -93.0158615, 92.9138031
11: -48.6770706, 21.4562340, -49.1740036, 21.9127254, -70.5897980, 70.6302338
12: -55.0643005, 24.7012367, -55.3461990, 25.1673298, -79.1096725, 78.8986588
13: -50.2556000, 43.5147247, -50.5680618, 43.8049088, -94.0605087, 94.0827866
14: -87.1413269, 30.5684471, -87.5766907, 31.1142597, -118.2555847, 118.1451416
15: -35.4728355, 35.9047127, -35.7802048, 36.0475082, -71.5203400, 71.6849213
16: -45.6333427, 33.5753784, -45.9141388, 33.8669281, -79.5002747, 79.4895172
17: -84.5398712, 23.0409832, -84.9580383, 23.4395351, -107.9794083, 107.9990234
18: -48.6691360, 30.9589996, -49.0731659, 31.2634964, -79.9326324, 80.0321655
19: -38.6457138, 18.2236519, -38.9918671, 18.5124054, -57.1581192, 57.2155190
20: -36.6911697, 23.0787163, -36.9594421, 23.3459320, -60.0371017, 60.0381584
21: -47.7994347, 21.6982975, -48.2153969, 22.0612259, -69.8606567, 69.9136963
22: -49.8522682, 21.9085083, -50.0452576, 22.1102676, -71.9625397, 71.9537659
23: -38.8548889, 23.6787300, -39.1484528, 23.9916592, -62.8465500, 62.8271828
24: -46.1007690, 23.7727108, -46.3574333, 23.9967079, -70.0974731, 70.1301422
25: -41.0520897, 24.5144348, -41.2732086, 24.7690125, -65.8211060, 65.7876434
26: -56.6846542, 32.9938545, -57.0416641, 33.5029984, -90.1876526, 90.0355225
27: -45.0874786, 28.5767441, -45.3281784, 28.7265816, -73.8140564, 73.9049225
28: -38.7362404, 26.4417152, -38.9834251, 26.7157898, -65.4520264, 65.4251404
29: -51.4711494, 20.3387527, -51.7128105, 20.5990124, -72.0701599, 72.0515594
30: -49.0360641, 25.7701073, -49.3493881, 26.1519814, -75.1880493, 75.1194916
31: -50.6907234, 27.4892120, -51.1162338, 27.8311825, -78.5219040, 78.6054459
32: -52.2238388, 24.5099678, -52.3805161, 24.6600571, -76.8838959, 76.8904877
33: -71.9545593, 33.5401154, -72.2782288, 33.8191986, -105.5287094, 105.7105103
34: -65.3313904, 16.9650459, -65.4903412, 17.1366024, -81.5931854, 81.7317581
35: -63.5729141, 23.3787842, -63.7504959, 23.5540409, -85.5583038, 85.6349182
36: -61.7853928, 24.2490921, -61.9539490, 24.3912315, -86.1766205, 86.2030411
37: -86.9618835, 19.6321373, -87.1413498, 19.8028183, -106.7647018, 106.7734833
38: -69.6679230, 29.0050278, -69.9130249, 29.2008667, -98.8687897, 98.9180527
39: -80.1268005, 30.4488373, -80.3959961, 30.6448746, -110.7716751, 110.8448334
40: -62.3787193, 25.5440731, -62.5538025, 25.6919193, -88.0706406, 88.0978775
41: -54.8115540, 32.6944237, -54.9736862, 32.8565979, -87.6681519, 87.6681061
42: -36.1192398, 25.7699280, -36.2150269, 25.9892006, -62.1084404, 61.9849548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1866825
time: 114.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2340991
time: 86.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -56.4724731, 43.4402275, -56.5283813, 43.3908386, -99.8633118, 99.9686127
1: -25.2109451, 37.6824112, -25.2595119, 37.6871490, -62.8980942, 62.9419250
2: -21.7298260, 37.1253853, -21.8060226, 37.1616745, -58.8915024, 58.9314079
3: -24.3229465, 39.7200546, -24.4720612, 39.8305779, -64.1535263, 64.1921158
4: -28.3703060, 43.6369324, -28.4457741, 43.6396637, -72.0099716, 72.0827026
5: -24.5522213, 39.6865692, -24.6552677, 39.7194481, -64.2716675, 64.3418350
6: -54.2116013, 31.7495289, -54.2516899, 31.7825508, -85.9941559, 86.0012207
7: -30.3420162, 39.5162506, -30.3914280, 39.4621811, -69.8041992, 69.9076767
8: -36.4947624, 53.4705963, -36.5544891, 53.5076828, -90.0024414, 90.0250854
9: -29.0327530, 38.9504051, -29.0805950, 38.9718475, -68.0046005, 68.0309982
10: -49.3544235, 43.5074463, -49.5499268, 43.7455139, -93.0999374, 93.0573730
11: -48.8525238, 21.7301903, -49.0282593, 21.9130592, -70.7655792, 70.7584534
12: -55.1498222, 24.9953747, -55.1736450, 25.1281357, -79.1472473, 79.0101624
13: -50.3829460, 43.6381416, -50.5304489, 43.7811279, -94.1640778, 94.1685944
14: -87.2752533, 30.8876019, -87.3950272, 31.0866203, -118.3618774, 118.2826309
15: -35.5937309, 35.9715805, -35.7164192, 36.0270462, -71.6207733, 71.6880035
16: -45.7514191, 33.6518974, -45.8701515, 33.7446289, -79.4960480, 79.5220490
17: -84.6611099, 23.2837696, -84.7197418, 23.4051094, -108.0662231, 108.0035095
18: -48.8398438, 31.1375065, -49.0762596, 31.3038445, -80.1436920, 80.2137680
19: -38.7813225, 18.4041901, -38.9068680, 18.5224705, -57.3037949, 57.3110580
20: -36.8032608, 23.2431984, -36.8822632, 23.3386021, -60.1418610, 60.1254616
21: -47.9539757, 21.9235420, -48.0779114, 22.0535202, -70.0074921, 70.0014496
22: -49.8778114, 22.0424271, -49.8092308, 22.0534687, -71.9312820, 71.8516541
23: -38.9499168, 23.8620796, -39.0256233, 23.9679871, -62.9179039, 62.8877029
24: -46.1811371, 23.9021816, -46.2560577, 23.9861069, -70.1672440, 70.1582413
25: -41.0983772, 24.6615543, -41.1261444, 24.7166748, -65.8150482, 65.7876968
26: -56.8209343, 33.3184738, -56.8799553, 33.4935265, -90.3144608, 90.1984253
27: -45.2235832, 28.6738529, -45.2961998, 28.7269726, -73.9505539, 73.9700546
28: -38.8435936, 26.6289806, -38.8851471, 26.6969261, -65.5405197, 65.5141296
29: -51.5050888, 20.5164642, -51.4315948, 20.5359612, -72.0410461, 71.9480591
30: -49.1356125, 25.9955826, -49.2019463, 26.1206741, -75.2562866, 75.1975250
31: -50.8530159, 27.6843739, -51.0367699, 27.8369293, -78.6899414, 78.7211456
32: -52.2956429, 24.6019058, -52.3186531, 24.6487846, -76.9444275, 76.9205627
33: -72.1375427, 33.6654015, -72.2655487, 33.7870636, -105.6723938, 105.7109451
34: -65.3914337, 17.0484104, -65.4556580, 17.0994949, -81.6882477, 81.7151108
35: -63.6227074, 23.4346752, -63.6902771, 23.5308495, -85.6779633, 85.6511612
36: -61.8286362, 24.2970276, -61.8922462, 24.3642578, -86.1928940, 86.1892700
37: -87.0415649, 19.7179642, -87.1056595, 19.7695675, -106.8111343, 106.8236237
38: -69.7609711, 29.0484657, -69.8825760, 29.1251183, -98.8860931, 98.9310455
39: -80.2559052, 30.5072575, -80.3470383, 30.5461388, -110.8020477, 110.8542938
40: -62.4609985, 25.5636024, -62.5056000, 25.5313034, -87.9923019, 88.0691986
41: -54.8810577, 32.7637558, -54.9342422, 32.7900734, -87.6711273, 87.6979980
42: -36.1653900, 25.8698559, -36.1836662, 25.9642754, -62.1296654, 62.0535202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 932

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1880775
time: 80.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
time: 99.41 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 182.16 seconds
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1826639
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2301780
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1861226
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2338611
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1880775
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1888820
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2363706
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1829501
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2302435
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1866825
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2340991
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.1880775
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2355222
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2352701
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -44.0636384, upper bound: 44.2366513
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2379575
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2352701
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -43.9697304, upper bound: 44.2314475
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 182.16
Output dim: 4, lower bound: -43.9994580, upper bound: 44.2379575

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 101.09 + 7117.22 = 7218.31 seconds
