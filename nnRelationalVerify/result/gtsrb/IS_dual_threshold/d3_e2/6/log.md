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
execution time: IAR + RelationalAnalysis = 2.96 + 100.72 = 103.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -44.2596780, upper bound: 44.2596780

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2545344, upper bound: 44.1713845
time: 91.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2545344, upper bound: 44.2545341
time: 114.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 205.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 205.58
Output dim: 4, lower bound: -44.2545344, upper bound: 44.1713845
IS_A2, status: Status.UNKNOWN, split count: 1, time: 205.58
Output dim: 4, lower bound: -44.2545344, upper bound: 44.2545341

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -56.5774345, 43.4741249, -56.7218208, 43.5520630, -100.1295013, 100.1959457
1: -25.3130684, 37.7537231, -25.4128113, 37.8272629, -63.1403313, 63.1665344
2: -21.8691521, 37.2188339, -21.9853630, 37.2735367, -59.1426888, 59.2041969
3: -24.5831909, 39.8767700, -24.6373005, 39.9255409, -64.5087280, 64.5140686
4: -28.5065613, 43.7128754, -28.6770439, 43.8198471, -72.3264084, 72.3899231
5: -24.7277393, 39.7589951, -24.8279839, 39.8436508, -64.5713882, 64.5869751
6: -54.2596397, 31.8471947, -54.3219757, 31.9237061, -86.1833496, 86.1691742
7: -30.4395580, 39.5252686, -30.5806351, 39.6276817, -70.0672379, 70.1059036
8: -36.6277008, 53.5948410, -36.7368317, 53.6800079, -90.3077087, 90.3316727
9: -29.1137161, 39.0345955, -29.1663876, 39.0972900, -68.2110062, 68.2009811
10: -49.5445023, 43.8179855, -49.6492615, 43.9876175, -93.5321198, 93.4672470
11: -49.0903015, 22.0186138, -49.2038765, 22.1462097, -71.2365112, 71.2224884
12: -55.2228088, 25.2224922, -55.3498268, 25.3977089, -79.5150681, 79.4624329
13: -50.6660728, 43.8019371, -50.7347984, 43.8663712, -94.5324402, 94.5367355
14: -87.3826752, 31.1273499, -87.5954514, 31.3981152, -118.7807922, 118.7228012
15: -35.8307114, 36.0474167, -35.9293556, 36.0916595, -71.9223709, 71.9767761
16: -45.9002571, 33.9075050, -46.0011406, 34.0331497, -79.9334106, 79.9086456
17: -84.7609177, 23.4591274, -84.9513550, 23.6389332, -108.3998489, 108.4104843
18: -49.0825119, 31.4745102, -49.1090393, 31.5206699, -80.6031799, 80.5835495
19: -38.9744644, 18.6264534, -39.0370598, 18.6576881, -57.6321526, 57.6635132
20: -36.9345016, 23.3967171, -37.0024109, 23.4771709, -60.4116745, 60.3991280
21: -48.1669540, 22.1658897, -48.2566910, 22.2425175, -70.4094696, 70.4225769
22: -49.9105148, 22.0828743, -50.0595703, 22.2043266, -72.1148376, 72.1424408
23: -39.0753975, 24.0364990, -39.1697922, 24.1438179, -63.2192154, 63.2062912
24: -46.2972183, 24.0283127, -46.3904076, 24.1069946, -70.4042130, 70.4187164
25: -41.1805954, 24.7483654, -41.2919159, 24.8896675, -66.0702667, 66.0402832
26: -56.9476852, 33.6187477, -57.0730629, 33.7571182, -90.7048035, 90.6918106
27: -45.3204155, 28.7819881, -45.3839912, 28.8251553, -74.1455688, 74.1659775
28: -38.9406433, 26.7513390, -39.0207405, 26.8487740, -65.7894135, 65.7720795
29: -51.5442085, 20.5690651, -51.7131500, 20.7256393, -72.2698517, 72.2822113
30: -49.2738762, 26.1912231, -49.3743286, 26.3364830, -75.6103592, 75.5655518
31: -51.0946617, 27.9382019, -51.1684227, 28.0036850, -79.0983429, 79.1066284
32: -52.3645668, 24.6795692, -52.4324188, 24.7214165, -77.0859833, 77.1119843
33: -72.3192139, 33.7754440, -72.4350433, 33.8517761, -105.8650665, 105.9955750
34: -65.5065918, 17.1340504, -65.5713196, 17.1903362, -81.8670273, 81.9528351
35: -63.7821579, 23.5537663, -63.8446465, 23.5854778, -85.8050461, 85.9005890
36: -61.9601021, 24.3872662, -62.0272026, 24.4306393, -86.3907394, 86.4144669
37: -87.0604248, 19.7939873, -87.1947250, 19.9000473, -106.9604721, 106.9887085
38: -69.9110413, 29.1409874, -70.0140991, 29.2295399, -99.1405792, 99.1550903
39: -80.3699493, 30.5510883, -80.5175171, 30.6526318, -111.0225830, 111.0686035
40: -62.4887352, 25.5804348, -62.6132050, 25.7111301, -88.1998672, 88.1936417
41: -54.9157143, 32.8210220, -55.0350189, 32.9084320, -87.8241425, 87.8560410
42: -36.2395592, 26.0187168, -36.2789688, 26.0683708, -62.3079300, 62.2976837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2475336, upper bound: 44.0891282
time: 88.65 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.2498324, upper bound: 44.1650555
time: 77.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -56.7303276, 43.5710983, -56.7349777, 43.5773849, -100.3077087, 100.3060760
1: -25.4170418, 37.8526840, -25.4200687, 37.8565598, -63.2736015, 63.2727509
2: -21.9902878, 37.2906036, -21.9936008, 37.2935219, -59.2838097, 59.2842026
3: -24.6384926, 39.9351616, -24.6412468, 39.9375839, -64.5760803, 64.5764084
4: -28.6850548, 43.8563957, -28.6886292, 43.8624268, -72.5474854, 72.5450287
5: -24.8312168, 39.8687820, -24.8343487, 39.8722534, -64.7034683, 64.7031326
6: -54.3261261, 31.9355354, -54.3286095, 31.9435005, -86.2696228, 86.2641449
7: -30.5867386, 39.6608963, -30.5909710, 39.6661682, -70.2529068, 70.2518692
8: -36.7398643, 53.7065048, -36.7432289, 53.7108231, -90.4506836, 90.4497375
9: -29.1706848, 39.0933914, -29.1732292, 39.1068382, -68.2775269, 68.2666168
10: -49.6788101, 43.9912643, -49.6839294, 43.9967957, -93.6756058, 93.6751938
11: -49.2354202, 22.1501732, -49.2427788, 22.1528301, -71.3882523, 71.3929520
12: -55.3912201, 25.3985214, -55.3969116, 25.4039364, -79.6768951, 79.6912384
13: -50.7370262, 43.8726692, -50.7433014, 43.8777618, -94.6147919, 94.6159668
14: -87.6620789, 31.3984852, -87.6712952, 31.4031029, -119.0651855, 119.0697784
15: -35.9321442, 36.0960960, -35.9459000, 36.0988617, -72.0310059, 72.0419922
16: -46.0086441, 34.0345764, -46.0135918, 34.0606079, -80.0692520, 80.0481720
17: -85.0147095, 23.6415977, -85.0226288, 23.6459427, -108.6606522, 108.6642303
18: -49.1223640, 31.4703007, -49.1261063, 31.5074310, -80.6297913, 80.5964050
19: -39.0480042, 18.6579742, -39.0507736, 18.6593208, -57.7073250, 57.7087479
20: -37.0157356, 23.4801846, -37.0188828, 23.4818554, -60.4975891, 60.4990692
21: -48.2777634, 22.2451019, -48.2817001, 22.2467422, -70.5245056, 70.5268021
22: -50.1043167, 22.2072792, -50.1126137, 22.2104874, -72.3148041, 72.3198929
23: -39.1976166, 24.1463509, -39.2015991, 24.1480656, -63.3456802, 63.3479500
24: -46.4128036, 24.1097546, -46.4174576, 24.1109142, -70.5237198, 70.5272141
25: -41.3265495, 24.8980103, -41.3315735, 24.9006252, -66.2271729, 66.2295837
26: -57.1055145, 33.7559204, -57.1114731, 33.7606087, -90.8661194, 90.8673935
27: -45.3903351, 28.8220329, -45.3942146, 28.8318291, -74.2221680, 74.2162476
28: -39.0370102, 26.8513317, -39.0420189, 26.8533039, -65.8903122, 65.8933487
29: -51.7661781, 20.7291260, -51.7751465, 20.7316914, -72.4978714, 72.5042725
30: -49.4039612, 26.3415565, -49.4097443, 26.3446560, -75.7486191, 75.7512970
31: -51.1837234, 28.0059471, -51.1875763, 28.0076103, -79.1913300, 79.1935272
32: -52.4394608, 24.7244396, -52.4451675, 24.7264786, -77.1659393, 77.1696091
33: -72.4427872, 33.8697815, -72.4460754, 33.8744583, -106.1371613, 106.0588531
34: -65.5651703, 17.1962070, -65.5790405, 17.1991119, -82.0280457, 81.9385681
35: -63.8470459, 23.5906315, -63.8531265, 23.5932579, -85.9836273, 85.9158630
36: -62.0326843, 24.4297504, -62.0360756, 24.4382553, -86.4709396, 86.4658279
37: -87.2059021, 19.9025688, -87.2098770, 19.9238834, -107.1297836, 107.1124420
38: -70.0199509, 29.2573586, -70.0243530, 29.2621956, -99.2821503, 99.2817078
39: -80.5246429, 30.6835899, -80.5293427, 30.6917877, -111.2164307, 111.2129364
40: -62.6196098, 25.7548313, -62.6236038, 25.7615662, -88.3811798, 88.3784332
41: -55.0402908, 32.9311790, -55.0433807, 32.9379730, -87.9782639, 87.9745636
42: -36.2629509, 26.0783424, -36.2803001, 26.0816021, -62.3445511, 62.3586426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1513619, upper bound: 44.2475332
time: 81.07 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.1513619, upper bound: 44.2498593
time: 106.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 190.24 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 190.24
Output dim: 4, lower bound: -44.2475336, upper bound: 44.0891282
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 190.24
Output dim: 4, lower bound: -44.2498324, upper bound: 44.1650555
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 190.24
Output dim: 4, lower bound: -44.1513619, upper bound: 44.2475332
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 190.24
Output dim: 4, lower bound: -44.1513619, upper bound: 44.2498593

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -56.4443741, 43.3188820, -56.7033882, 43.5008240, -99.9451981, 100.0222702
1: -25.2156124, 37.6045532, -25.4025345, 37.7763367, -62.9919510, 63.0070877
2: -21.7659588, 37.1067696, -21.9754734, 37.2355042, -59.0014648, 59.0822449
3: -24.5187435, 39.7958107, -24.6304474, 39.8997116, -64.4184570, 64.4262543
4: -28.3647404, 43.5134735, -28.6649742, 43.7518234, -72.1165619, 72.1784515
5: -24.6382637, 39.6312714, -24.8187256, 39.8017159, -64.4399796, 64.4499969
6: -54.2168579, 31.7449722, -54.3118858, 31.8930416, -86.1099014, 86.0568542
7: -30.3140297, 39.3447800, -30.5667896, 39.5659714, -69.8800049, 69.9115677
8: -36.5176773, 53.4227715, -36.7273254, 53.6213875, -90.1390686, 90.1501007
9: -29.0627728, 38.9467087, -29.1580639, 39.0717468, -68.1345215, 68.1047745
10: -49.4569931, 43.7254257, -49.6249084, 43.9704437, -93.4274368, 93.3503342
11: -48.9190445, 21.9225140, -49.1483917, 22.1365719, -71.0556183, 71.0709076
12: -55.0309486, 25.0804214, -55.2857361, 25.3864536, -79.3100128, 79.2487640
13: -50.5694275, 43.7488441, -50.7093658, 43.8517494, -94.4211731, 94.4582062
14: -87.1657410, 30.9865265, -87.5279160, 31.3895187, -118.5552597, 118.5144424
15: -35.7212601, 36.0091705, -35.9018402, 36.0813446, -71.8026047, 71.9110107
16: -45.8198395, 33.7233505, -45.9825325, 33.9773560, -79.7971954, 79.7058868
17: -84.4971161, 23.3381824, -84.8656616, 23.6274204, -108.1245346, 108.2038422
18: -49.0670013, 31.4157200, -49.0914764, 31.5046844, -80.5716858, 80.5071945
19: -38.8670425, 18.5742645, -39.0051270, 18.6529884, -57.5200310, 57.5793915
20: -36.8336220, 23.3317490, -36.9740982, 23.4715500, -60.3051720, 60.3058472
21: -48.0032616, 22.0793934, -48.2061386, 22.2358189, -70.2390823, 70.2855301
22: -49.6499901, 21.9793720, -49.9711876, 22.1968403, -71.8468323, 71.9505615
23: -38.9319992, 23.9494743, -39.1241531, 24.1375427, -63.0695419, 63.0736275
24: -46.1736374, 23.9728203, -46.3521996, 24.1016960, -70.2753296, 70.3250198
25: -41.0117493, 24.6429729, -41.2369385, 24.8797874, -65.8915405, 65.8799133
26: -56.7589340, 33.4962502, -57.0150337, 33.7488403, -90.5077744, 90.5112839
27: -45.2638893, 28.7379704, -45.3672867, 28.8128433, -74.0767365, 74.1052551
28: -38.8194962, 26.6674767, -38.9846039, 26.8429813, -65.6624756, 65.6520844
29: -51.2388954, 20.4436073, -51.6117592, 20.7192078, -71.9580994, 72.0553665
30: -49.1036453, 26.0806484, -49.3183861, 26.3261108, -75.4297562, 75.3990326
31: -50.9885139, 27.8758545, -51.1367645, 27.9969978, -78.9855118, 79.0126190
32: -52.2796440, 24.6379471, -52.4074860, 24.7137909, -76.9934387, 77.0454330
33: -72.2355118, 33.7202606, -72.4183350, 33.8386917, -105.7093658, 105.9142990
34: -65.4469299, 17.0735378, -65.5547562, 17.1778812, -81.7634430, 81.8654327
35: -63.6901093, 23.5157623, -63.8170280, 23.5745659, -85.6992645, 85.8510361
36: -61.8791275, 24.3428040, -62.0038757, 24.4215527, -86.3006821, 86.3466797
37: -87.0064850, 19.7120552, -87.1747284, 19.8776054, -106.8840942, 106.8867798
38: -69.8504486, 29.0457611, -69.9968338, 29.1969376, -99.0473862, 99.0425949
39: -80.2733383, 30.4347324, -80.4991760, 30.6148891, -110.8882294, 110.9339066
40: -62.4266129, 25.3985844, -62.5975952, 25.6479721, -88.0745850, 87.9961777
41: -54.8580170, 32.7225456, -55.0238800, 32.8790550, -87.7370758, 87.7464294
42: -36.1834106, 25.9637012, -36.2635498, 26.0529137, -62.2363243, 62.2272491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2030532, upper bound: 44.0891282
time: 93.43 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.2030532, upper bound: 44.0891282
time: 85.45 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -56.5728912, 43.4682770, -56.7207375, 43.5511017, -100.1239929, 100.1890106
1: -25.3106041, 37.7491302, -25.4122410, 37.8262329, -63.1368370, 63.1613693
2: -21.8665829, 37.2153435, -21.9847374, 37.2727699, -59.1393509, 59.2000809
3: -24.5811958, 39.8737068, -24.6368828, 39.9248047, -64.5059967, 64.5105896
4: -28.5033894, 43.7074814, -28.6762772, 43.8186226, -72.3220139, 72.3837585
5: -24.7248363, 39.7551880, -24.8273315, 39.8427811, -64.5676193, 64.5825195
6: -54.2574234, 31.8334408, -54.3214722, 31.9206619, -86.1780853, 86.1549149
7: -30.4357243, 39.5187988, -30.5797272, 39.6261826, -70.0619049, 70.0985260
8: -36.6252747, 53.5887794, -36.7362862, 53.6786385, -90.3039093, 90.3250656
9: -29.1123333, 39.0239029, -29.1660500, 39.0949097, -68.2072449, 68.1899567
10: -49.5368690, 43.8132477, -49.6476212, 43.9864845, -93.5233536, 93.4608688
11: -49.0849648, 22.0164337, -49.2026749, 22.1457138, -71.2306824, 71.2191086
12: -55.2167854, 25.2185249, -55.3484688, 25.3968372, -79.4973755, 79.4571075
13: -50.6504593, 43.7980881, -50.7313652, 43.8654785, -94.5159378, 94.5294495
14: -87.3757248, 31.1235161, -87.5938873, 31.3972340, -118.7729568, 118.7174072
15: -35.8187866, 36.0449409, -35.9266739, 36.0910492, -71.9098358, 71.9716187
16: -45.8969116, 33.8759499, -46.0003738, 34.0258255, -79.9227371, 79.8763275
17: -84.7534027, 23.4552441, -84.9497147, 23.6379356, -108.3913422, 108.4049606
18: -49.0782280, 31.4440193, -49.1079865, 31.5140152, -80.5922394, 80.5520020
19: -38.9709435, 18.6252766, -39.0362434, 18.6574173, -57.6283607, 57.6615219
20: -36.9313736, 23.3953457, -37.0016632, 23.4768448, -60.4082184, 60.3970108
21: -48.1619606, 22.1645851, -48.2555122, 22.2422256, -70.4041901, 70.4200974
22: -49.9006691, 22.0802155, -50.0571899, 22.2037048, -72.1043701, 72.1374054
23: -39.0715714, 24.0346985, -39.1689301, 24.1434155, -63.2149887, 63.2036285
24: -46.2917404, 24.0273418, -46.3890724, 24.1067734, -70.3985138, 70.4164124
25: -41.1748734, 24.7451935, -41.2906151, 24.8888550, -66.0637283, 66.0358124
26: -56.9410896, 33.6158066, -57.0715637, 33.7564926, -90.6975861, 90.6873703
27: -45.3165321, 28.7636642, -45.3829765, 28.8214188, -74.1379547, 74.1466370
28: -38.9374084, 26.7494354, -39.0200157, 26.8483372, -65.7857437, 65.7694550
29: -51.5334663, 20.5669365, -51.7105179, 20.7251339, -72.2585983, 72.2774506
30: -49.2681961, 26.1879444, -49.3730392, 26.3357353, -75.6039276, 75.5609818
31: -51.0909615, 27.9368210, -51.1675987, 28.0033817, -79.0943451, 79.1044159
32: -52.3559647, 24.6777039, -52.4304161, 24.7209892, -77.0769501, 77.1081238
33: -72.3084106, 33.7734756, -72.4326935, 33.8513184, -105.9137726, 105.9850769
34: -65.4998398, 17.1322556, -65.5698318, 17.1899300, -81.8855591, 81.9297714
35: -63.7667542, 23.5524540, -63.8412361, 23.5851574, -85.7943115, 85.8906937
36: -61.9523163, 24.3857403, -62.0253372, 24.4303017, -86.3826141, 86.4110794
37: -87.0560303, 19.7718983, -87.1936340, 19.8951511, -106.9511795, 106.9655304
38: -69.9058914, 29.1272697, -70.0129089, 29.2263603, -99.1322479, 99.1401825
39: -80.3650131, 30.5475368, -80.5163193, 30.6518764, -111.0168915, 111.0638580
40: -62.4844589, 25.5561485, -62.6121140, 25.7056808, -88.1901398, 88.1682587
41: -54.9134903, 32.8048401, -55.0345001, 32.9050560, -87.8185425, 87.8393402
42: -36.2292023, 26.0154648, -36.2767258, 26.0675621, -62.2967644, 62.2921906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=229, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=486, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 632

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1513371, upper bound: 44.1627844
time: 86.91 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.1513371, upper bound: 44.1650559
time: 81.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -56.7118378, 43.5190849, -56.6003342, 43.4191818, -100.1310196, 100.1194153
1: -25.4067631, 37.8017426, -25.3217335, 37.7067490, -63.1135101, 63.1234741
2: -21.9804153, 37.2525673, -21.8890934, 37.1809769, -59.1613922, 59.1416626
3: -24.6316528, 39.9093323, -24.5755577, 39.8556442, -64.4872971, 64.4848938
4: -28.6730042, 43.7883301, -28.5448742, 43.6623535, -72.3353577, 72.3332062
5: -24.8219795, 39.8268280, -24.7437439, 39.7439117, -64.5658875, 64.5705719
6: -54.3160362, 31.9048023, -54.2835236, 31.8364525, -86.1524887, 86.1883240
7: -30.5729160, 39.5991440, -30.4641647, 39.4846725, -70.0575867, 70.0633087
8: -36.7303810, 53.6478729, -36.6318970, 53.5376663, -90.2680511, 90.2797699
9: -29.1624069, 39.0678482, -29.1205330, 39.0175476, -68.1799545, 68.1883850
10: -49.6545029, 43.9739952, -49.5955200, 43.9017715, -93.5562744, 93.5695190
11: -49.1799316, 22.1405029, -49.0703735, 22.0550079, -71.2349396, 71.2108765
12: -55.3271332, 25.3873348, -55.2039337, 25.2603455, -79.4601135, 79.4850616
13: -50.7117233, 43.8580589, -50.6434860, 43.8231087, -94.5348358, 94.5015411
14: -87.5943756, 31.3899574, -87.4528961, 31.2600842, -118.8544617, 118.8428497
15: -35.9042816, 36.0857773, -35.8302460, 36.0584831, -71.9627686, 71.9160233
16: -45.9901123, 33.9787369, -45.9316216, 33.8719749, -79.8620911, 79.9103546
17: -84.9289551, 23.6300755, -84.7577591, 23.5227203, -108.4516754, 108.3878326
18: -49.1048431, 31.4543858, -49.1087189, 31.4471703, -80.5520172, 80.5631027
19: -39.0160751, 18.6532593, -38.9421539, 18.6063538, -57.6224289, 57.5954132
20: -36.9873886, 23.4746056, -36.9171524, 23.4163437, -60.4037323, 60.3917580
21: -48.2271729, 22.2384052, -48.1166458, 22.1594753, -70.3866501, 70.3550491
22: -50.0157051, 22.1998100, -49.8496552, 22.1057930, -72.1214981, 72.0494690
23: -39.1519699, 24.1400394, -39.0576172, 24.0600567, -63.2120285, 63.1976547
24: -46.3744278, 24.1044884, -46.2930679, 24.0547943, -70.4292221, 70.3975525
25: -41.2715607, 24.8881512, -41.1619110, 24.7934227, -66.0649872, 66.0500641
26: -57.0474586, 33.7477112, -56.9214020, 33.6367798, -90.6842346, 90.6691132
27: -45.3735504, 28.8084641, -45.3361969, 28.7834435, -74.1569977, 74.1446609
28: -39.0008850, 26.8455563, -38.9203568, 26.7684784, -65.7693634, 65.7659149
29: -51.6644592, 20.7226791, -51.4681206, 20.6051292, -72.2695923, 72.1907959
30: -49.3479691, 26.3311996, -49.2386322, 26.2327805, -75.5807495, 75.5698318
31: -51.1520767, 27.9992371, -51.0804024, 27.9444466, -79.0965271, 79.0796356
32: -52.4135132, 24.7168198, -52.3580055, 24.6840363, -77.0975494, 77.0748291
33: -72.4261017, 33.8567200, -72.3604889, 33.8186035, -106.0552368, 105.9008331
34: -65.5483551, 17.1837521, -65.5173874, 17.1358757, -81.9407959, 81.8330765
35: -63.8195114, 23.5797749, -63.7583580, 23.5546856, -85.9334106, 85.8084793
36: -62.0093193, 24.4206810, -61.9533348, 24.3934422, -86.4027634, 86.3740158
37: -87.1858368, 19.8784008, -87.1492310, 19.8399353, -107.0257721, 107.0276337
38: -70.0027237, 29.2241879, -69.9594345, 29.1590157, -99.1617432, 99.1836243
39: -80.5062714, 30.6458321, -80.4301758, 30.5748329, -111.0811005, 111.0760040
40: -62.6039162, 25.6898766, -62.5490417, 25.5678654, -88.1717834, 88.2389221
41: -55.0291824, 32.9001579, -54.9803352, 32.8349113, -87.8640900, 87.8804932
42: -36.2475281, 26.0628757, -36.2231140, 26.0216637, -62.2691917, 62.2859879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2475329
time: 82.94 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2475335
time: 85.92 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -56.7293129, 43.5701141, -56.7306900, 43.5730133, -100.3023224, 100.3008041
1: -25.4165039, 37.8516464, -25.4176559, 37.8519707, -63.2684746, 63.2693024
2: -21.9897079, 37.2898407, -21.9911556, 37.2900047, -59.2797127, 59.2809982
3: -24.6380444, 39.9344292, -24.6392803, 39.9345093, -64.5725555, 64.5737076
4: -28.6842709, 43.8551979, -28.6855373, 43.8570862, -72.5413589, 72.5407333
5: -24.8305378, 39.8679352, -24.8315144, 39.8684616, -64.6989975, 64.6994476
6: -54.3256149, 31.9324951, -54.3264160, 31.9296703, -86.2552872, 86.2589111
7: -30.5858536, 39.6594467, -30.5872002, 39.6598053, -70.2456589, 70.2466431
8: -36.7393341, 53.7051544, -36.7408447, 53.7047958, -90.4441299, 90.4459991
9: -29.1703606, 39.0910263, -29.1717987, 39.0962639, -68.2666245, 68.2628250
10: -49.6771240, 43.9901886, -49.6763000, 43.9923401, -93.6694641, 93.6664886
11: -49.2342148, 22.1496658, -49.2373619, 22.1507893, -71.3850021, 71.3870239
12: -55.3898659, 25.3976402, -55.3908844, 25.4000435, -79.6716537, 79.6735840
13: -50.7336121, 43.8717804, -50.7276230, 43.8739243, -94.6075363, 94.5994034
14: -87.6605072, 31.3976631, -87.6643982, 31.3993244, -119.0598297, 119.0620575
15: -35.9296112, 36.0954895, -35.9338150, 36.0964203, -72.0260315, 72.0293045
16: -46.0078850, 34.0275726, -46.0102158, 34.0300980, -80.0379791, 80.0377884
17: -85.0130539, 23.6407337, -85.0151367, 23.6423340, -108.6553879, 108.6558685
18: -49.1213074, 31.4636593, -49.1218452, 31.4768429, -80.5981522, 80.5855026
19: -39.0471840, 18.6577339, -39.0472412, 18.6582375, -57.7054214, 57.7049751
20: -37.0150146, 23.4798851, -37.0158081, 23.4804897, -60.4955063, 60.4956932
21: -48.2766190, 22.2448292, -48.2768097, 22.2454739, -70.5220947, 70.5216370
22: -50.1020813, 22.2066307, -50.1031418, 22.2078342, -72.3099136, 72.3097687
23: -39.1967812, 24.1459351, -39.1977386, 24.1463013, -63.3430824, 63.3436737
24: -46.4115677, 24.1095257, -46.4123421, 24.1099854, -70.5215530, 70.5218658
25: -41.3252716, 24.8972130, -41.3259354, 24.8975067, -66.2227783, 66.2231445
26: -57.1040192, 33.7552948, -57.1048584, 33.7576981, -90.8617172, 90.8601532
27: -45.3894463, 28.8183498, -45.3906250, 28.8159599, -74.2054062, 74.2089767
28: -39.0363007, 26.8509007, -39.0388336, 26.8514366, -65.8877411, 65.8897324
29: -51.7637444, 20.7286015, -51.7650452, 20.7295990, -72.4933472, 72.4936447
30: -49.4027023, 26.3408356, -49.4042397, 26.3414516, -75.7441559, 75.7450714
31: -51.1829109, 28.0056381, -51.1838913, 28.0062981, -79.1892090, 79.1895294
32: -52.4377975, 24.7240086, -52.4378548, 24.7245903, -77.1623840, 77.1618652
33: -72.4403915, 33.8693542, -72.4353027, 33.8725471, -106.1267548, 106.1076126
34: -65.5637589, 17.1958046, -65.5725937, 17.1973515, -82.0050049, 81.9615402
35: -63.8436546, 23.5903225, -63.8377075, 23.5919456, -85.9737320, 85.9079971
36: -62.0308418, 24.4294147, -62.0283852, 24.4367390, -86.4675827, 86.4578018
37: -87.2048492, 19.8982220, -87.2056427, 19.9028797, -107.1077271, 107.1038666
38: -70.0187836, 29.2544556, -70.0192490, 29.2493629, -99.2681427, 99.2737045
39: -80.5235062, 30.6828079, -80.5246429, 30.6882324, -111.2117386, 111.2074509
40: -62.6186028, 25.7498398, -62.6195755, 25.7410278, -88.3596344, 88.3694153
41: -55.0397568, 32.9282112, -55.0411797, 32.9244766, -87.9642334, 87.9693909
42: -36.2606926, 26.0775509, -36.2699890, 26.0783710, -62.3390656, 62.3475418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=228, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498317
time: 105.35 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498596
time: 63.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 171.25 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.2030532, upper bound: 44.0891282
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.2030532, upper bound: 44.0891282
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.1513371, upper bound: 44.1627844
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.1513371, upper bound: 44.1650559
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2475329
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2475335
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498317
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 171.25
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498596

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -56.7118378, 43.5190849, -56.4444466, 43.3189735, -100.0308075, 99.9635315
1: -25.4067631, 37.8017426, -25.2156181, 37.6046906, -63.0114517, 63.0173607
2: -21.9804153, 37.2525673, -21.7659817, 37.1069489, -59.0873642, 59.0185471
3: -24.6316528, 39.9093323, -24.5187798, 39.7959824, -64.4276352, 64.4281158
4: -28.6730042, 43.7883301, -28.3647575, 43.5137634, -72.1867676, 72.1530914
5: -24.8219795, 39.8268280, -24.6382942, 39.6313705, -64.4533539, 64.4651184
6: -54.3160362, 31.9048023, -54.2172241, 31.7450066, -86.0610428, 86.1220245
7: -30.5729160, 39.5991440, -30.3140564, 39.3449135, -69.9178314, 69.9132004
8: -36.7303810, 53.6478729, -36.5177116, 53.4230423, -90.1534271, 90.1655884
9: -29.1624069, 39.0678482, -29.0627880, 38.9467430, -68.1091461, 68.1306381
10: -49.6545029, 43.9739952, -49.4570389, 43.7255554, -93.3800583, 93.4310303
11: -49.1799316, 22.1405029, -48.9191895, 21.9225311, -71.1024628, 71.0596924
12: -55.3271332, 25.3873348, -55.0314407, 25.0804634, -79.2907562, 79.3117065
13: -50.7117233, 43.8580589, -50.5694885, 43.7489014, -94.4606247, 94.4275513
14: -87.5943756, 31.3899574, -87.1659241, 30.9865780, -118.5809555, 118.5558777
15: -35.9042816, 36.0857773, -35.7212982, 36.0092468, -71.9135284, 71.8070755
16: -45.9901123, 33.9787369, -45.8198776, 33.7234726, -79.7135849, 79.7986145
17: -84.9289551, 23.6300755, -84.4971924, 23.3382378, -108.2671967, 108.1272659
18: -49.1048431, 31.4543858, -49.0671425, 31.4157600, -80.5205994, 80.5215302
19: -39.0160751, 18.6532593, -38.8671722, 18.5742836, -57.5903587, 57.5204315
20: -36.9873886, 23.4746056, -36.8337936, 23.3317528, -60.3191414, 60.3083992
21: -48.2271729, 22.2384052, -48.0035095, 22.0794258, -70.3065948, 70.2419128
22: -50.0157051, 22.1998100, -49.6501694, 21.9793873, -71.9950943, 71.8499756
23: -39.1519699, 24.1400394, -38.9321060, 23.9495049, -63.1014748, 63.0721436
24: -46.3744278, 24.1044884, -46.1736908, 23.9728470, -70.3472748, 70.2781830
25: -41.2715607, 24.8881512, -41.0118599, 24.6430130, -65.9145737, 65.9000092
26: -57.0474586, 33.7477112, -56.7591896, 33.4962769, -90.5437317, 90.5068970
27: -45.3735504, 28.8084641, -45.2641182, 28.7379818, -74.1115341, 74.0725861
28: -39.0008850, 26.8455563, -38.8195801, 26.6674995, -65.6683807, 65.6651382
29: -51.6644592, 20.7226791, -51.2391434, 20.4436188, -72.1080780, 71.9618225
30: -49.3479691, 26.3311996, -49.1037140, 26.0807800, -75.4287491, 75.4349136
31: -51.1520767, 27.9992371, -50.9886856, 27.8759003, -79.0279770, 78.9879227
32: -52.4135132, 24.7168198, -52.2801666, 24.6380024, -77.0515137, 76.9969864
33: -72.4261017, 33.8567200, -72.2355957, 33.7202950, -105.8768463, 105.7402039
34: -65.5483551, 17.1837521, -65.4470139, 17.0737553, -81.8116608, 81.7592087
35: -63.8195114, 23.5797749, -63.6901703, 23.5157890, -85.8147430, 85.7180176
36: -62.0093193, 24.4206810, -61.8796043, 24.3428345, -86.3521576, 86.3002853
37: -87.1858368, 19.8784008, -87.0067749, 19.7120838, -106.8979187, 106.8851776
38: -70.0027237, 29.2241879, -69.8509979, 29.0457935, -99.0485153, 99.0751877
39: -80.5062714, 30.6458321, -80.2735901, 30.4347763, -110.9410477, 110.9194183
40: -62.6039162, 25.6898766, -62.4267960, 25.3985977, -88.0025177, 88.1166687
41: -55.0291824, 32.9001579, -54.8582077, 32.7225838, -87.7517700, 87.7583618
42: -36.2475281, 26.0628757, -36.1837616, 25.9637413, -62.2112694, 62.2466354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=227, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 615

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2360989
time: 79.24 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2360990
time: 88.67 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -56.7118378, 43.5190849, -56.5955887, 43.4123688, -100.1242065, 100.1146698
1: -25.4067631, 37.8017426, -25.3186569, 37.7028961, -63.1096573, 63.1203995
2: -21.9804153, 37.2525673, -21.8856964, 37.1780663, -59.1584816, 59.1382637
3: -24.6316528, 39.9093323, -24.5727806, 39.8532257, -64.4848785, 64.4821167
4: -28.6730042, 43.7883301, -28.5411968, 43.6562843, -72.3292847, 72.3295288
5: -24.8219795, 39.8268280, -24.7405758, 39.7404480, -64.5624237, 64.5674057
6: -54.3160362, 31.9048023, -54.2810631, 31.8286228, -86.1446609, 86.1858673
7: -30.5729160, 39.5991440, -30.4598713, 39.4793587, -70.0522766, 70.0590134
8: -36.7303810, 53.6478729, -36.6284943, 53.5333710, -90.2637482, 90.2763672
9: -29.1624069, 39.0678482, -29.1180496, 39.0041046, -68.1665115, 68.1858978
10: -49.6545029, 43.9739952, -49.5904427, 43.8960571, -93.5505600, 93.5644379
11: -49.1799316, 22.1405029, -49.0630417, 22.0522232, -71.2321548, 71.2035446
12: -55.3271332, 25.3873348, -55.1982460, 25.2548714, -79.4546814, 79.4653015
13: -50.7117233, 43.8580589, -50.6375275, 43.8180695, -94.5297928, 94.4955902
14: -87.5943756, 31.3899574, -87.4436188, 31.2554512, -118.8498230, 118.8335724
15: -35.9042816, 36.0857773, -35.8164215, 36.0557289, -71.9600067, 71.9021988
16: -45.9901123, 33.9787369, -45.9267578, 33.8460312, -79.8361435, 79.9054947
17: -84.9289551, 23.6300755, -84.7498322, 23.5183067, -108.4472656, 108.3799057
18: -49.1048431, 31.4543858, -49.1049728, 31.4102535, -80.5150986, 80.5593567
19: -39.0160751, 18.6532593, -38.9393845, 18.6049423, -57.6210175, 57.5926437
20: -36.9873886, 23.4746056, -36.9139977, 23.4146385, -60.4020271, 60.3886032
21: -48.2271729, 22.2384052, -48.1126595, 22.1578007, -70.3849716, 70.3510666
22: -50.0157051, 22.1998100, -49.8411980, 22.1026306, -72.1183319, 72.0410080
23: -39.1519699, 24.1400394, -39.0536652, 24.0582962, -63.2102661, 63.1937027
24: -46.3744278, 24.1044884, -46.2882652, 24.0536251, -70.4280548, 70.3927536
25: -41.2715607, 24.8881512, -41.1568527, 24.7907867, -66.0623474, 66.0450058
26: -57.0474586, 33.7477112, -56.9154587, 33.6321335, -90.6795959, 90.6631699
27: -45.3735504, 28.8084641, -45.3321953, 28.7721291, -74.1456757, 74.1406555
28: -39.0008850, 26.8455563, -38.9153519, 26.7664528, -65.7673340, 65.7609100
29: -51.6644592, 20.7226791, -51.4588814, 20.6025257, -72.2669830, 72.1815643
30: -49.3479691, 26.3311996, -49.2328110, 26.2296028, -75.5775757, 75.5640106
31: -51.1520767, 27.9992371, -51.0765877, 27.9427376, -79.0948181, 79.0758209
32: -52.4135132, 24.7168198, -52.3506012, 24.6819611, -77.0954742, 77.0674210
33: -72.4261017, 33.8567200, -72.3572083, 33.8139114, -106.0498199, 105.9727173
34: -65.5483551, 17.1837521, -65.5030365, 17.1329651, -81.9347229, 81.9132156
35: -63.8195114, 23.5797749, -63.7524643, 23.5520344, -85.9298019, 85.8726425
36: -62.0093193, 24.4206810, -61.9498863, 24.3849068, -86.3942261, 86.3705673
37: -87.1858368, 19.8784008, -87.1452026, 19.8156395, -107.0014801, 107.0236053
38: -70.0027237, 29.2241879, -69.9550018, 29.1542530, -99.1569748, 99.1791916
39: -80.5062714, 30.6458321, -80.4254150, 30.5665989, -111.0728683, 111.0712433
40: -62.6039162, 25.6898766, -62.5449905, 25.5593948, -88.1633148, 88.2348633
41: -55.0291824, 32.9001579, -54.9772491, 32.8260345, -87.8552170, 87.8774109
42: -36.2475281, 26.0628757, -36.2057495, 26.0184631, -62.2659912, 62.2686234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 615

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2361448
time: 106.56 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2361448
time: 72.24 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -56.7293129, 43.5701141, -56.5728912, 43.4682770, -100.1975861, 100.1430054
1: -25.4165039, 37.8516464, -25.3106041, 37.7491302, -63.1656342, 63.1622505
2: -21.9897079, 37.2898407, -21.8665829, 37.2153435, -59.2050514, 59.1564255
3: -24.6380444, 39.9344292, -24.5811958, 39.8737068, -64.5117493, 64.5156250
4: -28.6842709, 43.8551979, -28.5033894, 43.7074814, -72.3917542, 72.3585892
5: -24.8305378, 39.8679352, -24.7248363, 39.7551880, -64.5857239, 64.5927734
6: -54.3256149, 31.9324951, -54.2574234, 31.8334408, -86.1590576, 86.1899185
7: -30.5858536, 39.6594467, -30.4357243, 39.5187988, -70.1046524, 70.0951691
8: -36.7393341, 53.7051544, -36.6252747, 53.5887794, -90.3281097, 90.3304291
9: -29.1703606, 39.0910263, -29.1123333, 39.0239029, -68.1942596, 68.2033615
10: -49.6771240, 43.9901886, -49.5368690, 43.8132477, -93.4903717, 93.5270538
11: -49.2342148, 22.1496658, -49.0849648, 22.0164337, -71.2506485, 71.2346344
12: -55.3898659, 25.3976402, -55.2167854, 25.2185249, -79.4990845, 79.4984894
13: -50.7336121, 43.8717804, -50.6504593, 43.7980881, -94.5317001, 94.5222397
14: -87.6605072, 31.3976631, -87.3757248, 31.1235161, -118.7840271, 118.7733917
15: -35.9296112, 36.0954895, -35.8187866, 36.0449409, -71.9745483, 71.9142761
16: -46.0078850, 34.0275726, -45.8969116, 33.8759499, -79.8838348, 79.9244843
17: -85.0130539, 23.6407337, -84.7534027, 23.4552441, -108.4682999, 108.3941345
18: -49.1213074, 31.4636593, -49.0782280, 31.4440193, -80.5653229, 80.5418854
19: -39.0471840, 18.6577339, -38.9709435, 18.6252766, -57.6724625, 57.6286774
20: -37.0150146, 23.4798851, -36.9313736, 23.3953457, -60.4103622, 60.4112587
21: -48.2766190, 22.2448292, -48.1619606, 22.1645851, -70.4412079, 70.4067917
22: -50.1020813, 22.2066307, -49.9006691, 22.0802155, -72.1822968, 72.1072998
23: -39.1967812, 24.1459351, -39.0715714, 24.0346985, -63.2314796, 63.2175064
24: -46.4115677, 24.1095257, -46.2917404, 24.0273418, -70.4389114, 70.4012680
25: -41.3252716, 24.8972130, -41.1748734, 24.7451935, -66.0704651, 66.0720825
26: -57.1040192, 33.7552948, -56.9410896, 33.6158066, -90.7198257, 90.6963806
27: -45.3894463, 28.8183498, -45.3165321, 28.7636642, -74.1531067, 74.1348801
28: -39.0363007, 26.8509007, -38.9374084, 26.7494354, -65.7857361, 65.7883072
29: -51.7637444, 20.7286015, -51.5334663, 20.5669365, -72.3306808, 72.2620697
30: -49.4027023, 26.3408356, -49.2681961, 26.1879444, -75.5906448, 75.6090317
31: -51.1829109, 28.0056381, -51.0909615, 27.9368210, -79.1197357, 79.0966034
32: -52.4377975, 24.7240086, -52.3559647, 24.6777039, -77.1155014, 77.0799713
33: -72.4403915, 33.8693542, -72.3084106, 33.7734756, -105.9474945, 105.9447937
34: -65.5637589, 17.1958046, -65.4998398, 17.1322556, -81.8775787, 81.8869095
35: -63.8436546, 23.5903225, -63.7667542, 23.5524540, -85.8578262, 85.8154907
36: -62.0308418, 24.4294147, -61.9523163, 24.3857403, -86.4165802, 86.3817291
37: -87.2048492, 19.8982220, -87.0560303, 19.7718983, -106.9767456, 106.9542542
38: -70.0187836, 29.2544556, -69.9058914, 29.1272697, -99.1460571, 99.1603470
39: -80.5235062, 30.6828079, -80.3650131, 30.5475368, -111.0710449, 111.0478210
40: -62.6186028, 25.7498398, -62.4844589, 25.5561485, -88.1747513, 88.2342987
41: -55.0397568, 32.9282112, -54.9134903, 32.8048401, -87.8445969, 87.8417053
42: -36.2606926, 26.0775509, -36.2292023, 26.0154648, -62.2761574, 62.3067551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=227, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0891282, upper bound: 44.1513365
time: 85.99 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498317
time: 71.47 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -56.7293129, 43.5701141, -56.7261810, 43.5667267, -100.2960358, 100.2962952
1: -25.4165039, 37.8516464, -25.4146233, 37.8481140, -63.2646179, 63.2662697
2: -21.9897079, 37.2898407, -21.9878807, 37.2870941, -59.2768021, 59.2777214
3: -24.6380444, 39.9344292, -24.6365662, 39.9320984, -64.5701447, 64.5709991
4: -28.6842709, 43.8551979, -28.6819763, 43.8510666, -72.5353394, 72.5371704
5: -24.8305378, 39.8679352, -24.8284073, 39.8650436, -64.6955795, 64.6963425
6: -54.3256149, 31.9324951, -54.3239365, 31.9218349, -86.2474518, 86.2564316
7: -30.5858536, 39.6594467, -30.5830002, 39.6546021, -70.2404556, 70.2424469
8: -36.7393341, 53.7051544, -36.7375145, 53.7005386, -90.4398727, 90.4426727
9: -29.1703606, 39.0910263, -29.1692905, 39.0828934, -68.2532501, 68.2603149
10: -49.6771240, 43.9901886, -49.6711693, 43.9869232, -93.6640472, 93.6613617
11: -49.2342148, 22.1496658, -49.2300034, 22.1481342, -71.3823471, 71.3796692
12: -55.3898659, 25.3976402, -55.3852463, 25.3946457, -79.6662903, 79.6541443
13: -50.7336121, 43.8717804, -50.7213821, 43.8688736, -94.6024857, 94.5931625
14: -87.6605072, 31.3976631, -87.6552658, 31.3947296, -119.0552368, 119.0529327
15: -35.9296112, 36.0954895, -35.9208069, 36.0936661, -72.0232773, 72.0162964
16: -46.0078850, 34.0275726, -46.0053177, 34.0045853, -80.0124664, 80.0328903
17: -85.0130539, 23.6407337, -85.0072556, 23.6380692, -108.6511230, 108.6479874
18: -49.1213074, 31.4636593, -49.1181335, 31.4397640, -80.5610733, 80.5817947
19: -39.0471840, 18.6577339, -39.0444946, 18.6569519, -57.7041359, 57.7022285
20: -37.0150146, 23.4798851, -37.0126991, 23.4788189, -60.4938354, 60.4925842
21: -48.2766190, 22.2448292, -48.2729263, 22.2438297, -70.5204468, 70.5177536
22: -50.1020813, 22.2066307, -50.0950737, 22.2046642, -72.3067474, 72.3017044
23: -39.1967812, 24.1459351, -39.1937714, 24.1445808, -63.3413620, 63.3397064
24: -46.4115677, 24.1095257, -46.4078598, 24.1088161, -70.5203857, 70.5173874
25: -41.3252716, 24.8972130, -41.3209534, 24.8949184, -66.2201920, 66.2181702
26: -57.1040192, 33.7552948, -57.0989380, 33.7530136, -90.8570328, 90.8542328
27: -45.3894463, 28.8183498, -45.3869171, 28.8061600, -74.1956024, 74.2052689
28: -39.0363007, 26.8509007, -39.0338364, 26.8494759, -65.8857727, 65.8847351
29: -51.7637444, 20.7286015, -51.7563705, 20.7270565, -72.4907990, 72.4849701
30: -49.4027023, 26.3408356, -49.3985786, 26.3384037, -75.7411041, 75.7394104
31: -51.1829109, 28.0056381, -51.1800880, 28.0046825, -79.1875916, 79.1857300
32: -52.4377975, 24.7240086, -52.4325218, 24.7225780, -77.1603775, 77.1565323
33: -72.4403915, 33.8693542, -72.4320679, 33.8678513, -106.1213684, 106.1830063
34: -65.5637589, 17.1958046, -65.5592270, 17.1944389, -81.9990234, 82.0620728
35: -63.8436546, 23.5903225, -63.8316727, 23.5893116, -85.9701691, 85.9742126
36: -62.0308418, 24.4294147, -62.0250397, 24.4282570, -86.4590988, 86.4544525
37: -87.2048492, 19.8982220, -87.2017822, 19.8828888, -107.0877380, 107.1000061
38: -70.0187836, 29.2544556, -70.0148621, 29.2445507, -99.2633362, 99.2693176
39: -80.5235062, 30.6828079, -80.5200348, 30.6800499, -111.2035522, 111.2028427
40: -62.6186028, 25.7498398, -62.6156960, 25.7342529, -88.3528595, 88.3655396
41: -55.0397568, 32.9282112, -55.0380630, 32.9176598, -87.9574127, 87.9662781
42: -36.2606926, 26.0775509, -36.2526360, 26.0751705, -62.3358612, 62.3301849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0891282, upper bound: 44.1513619
time: 76.77 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498596
time: 84.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 163.87 seconds
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2360989
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2360990
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2361448
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2361448
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0891282, upper bound: 44.1513365
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498317
IS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0891282, upper bound: 44.1513619
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 163.87
Output dim: 4, lower bound: -44.0891282, upper bound: 44.2498596

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -56.7069016, 43.5079575, -56.4229202, 43.2730064, -99.9799042, 99.9308777
1: -25.4042320, 37.7924843, -25.2042351, 37.5624695, -62.9667015, 62.9967194
2: -21.9779205, 37.2462883, -21.7554684, 37.0791359, -59.0570564, 59.0017548
3: -24.6279335, 39.9069405, -24.5023136, 39.7863884, -64.4143219, 64.4092560
4: -28.6696434, 43.7738533, -28.3513889, 43.4507713, -72.1204147, 72.1252441
5: -24.8198071, 39.8177261, -24.6290417, 39.5903130, -64.4101181, 64.4467697
6: -54.3136902, 31.8963623, -54.2071915, 31.7084236, -86.0221100, 86.1035538
7: -30.5699444, 39.5887299, -30.3013744, 39.2967911, -69.8667374, 69.8901062
8: -36.7288704, 53.6368942, -36.5114212, 53.3729935, -90.1018677, 90.1483154
9: -29.1597939, 39.0626831, -29.0519180, 38.9239120, -68.0837097, 68.1146011
10: -49.6447525, 43.9708023, -49.4150314, 43.7120972, -93.3568497, 93.3858337
11: -49.1652603, 22.1388226, -48.8518257, 21.9156837, -71.0809479, 70.9906464
12: -55.3196716, 25.3854122, -54.9986305, 25.0725574, -79.2707138, 79.2675247
13: -50.7095413, 43.8541031, -50.5607910, 43.7326508, -94.4421921, 94.4148941
14: -87.5806351, 31.3878632, -87.1025009, 30.9780712, -118.5587082, 118.4903641
15: -35.9006271, 36.0820923, -35.7081451, 35.9911880, -71.8918152, 71.7902374
16: -45.9853897, 33.9685287, -45.7988892, 33.6776123, -79.6630020, 79.7674179
17: -84.9118958, 23.6275787, -84.4180756, 23.3281097, -108.2400055, 108.0456543
18: -49.0996857, 31.4448433, -49.0474472, 31.3747063, -80.4743958, 80.4922943
19: -39.0085602, 18.6524620, -38.8332100, 18.5707989, -57.5793610, 57.4856720
20: -36.9824791, 23.4731560, -36.8122253, 23.3254128, -60.3078918, 60.2853813
21: -48.2164268, 22.2371330, -47.9560852, 22.0738411, -70.2902679, 70.1932220
22: -49.9990692, 22.1979427, -49.5733757, 21.9718113, -71.9708786, 71.7713165
23: -39.1385040, 24.1385689, -38.8699303, 23.9431076, -63.0816116, 63.0084991
24: -46.3639946, 24.1031227, -46.1267090, 23.9668655, -70.3308563, 70.2298279
25: -41.2574310, 24.8849163, -40.9464569, 24.6297665, -65.8871994, 65.8313751
26: -57.0420227, 33.7465897, -56.7352753, 33.4915657, -90.5335846, 90.4818649
27: -45.3700409, 28.8070488, -45.2495384, 28.7324467, -74.1024857, 74.0565872
28: -38.9900665, 26.8441715, -38.7708549, 26.6617470, -65.6518097, 65.6150284
29: -51.6427460, 20.7207241, -51.1387138, 20.4358063, -72.0785522, 71.8594360
30: -49.3305130, 26.3288403, -49.0291977, 26.0707340, -75.4012451, 75.3580399
31: -51.1442261, 27.9979534, -50.9528542, 27.8702621, -79.0144882, 78.9508057
32: -52.4095917, 24.7153988, -52.2663651, 24.6317902, -77.0413818, 76.9817657
33: -72.4225235, 33.8510742, -72.2200699, 33.6944656, -105.8184433, 105.7071991
34: -65.5406799, 17.1814156, -65.4137726, 17.0631180, -81.7768555, 81.7290955
35: -63.8137474, 23.5779934, -63.6655922, 23.5077095, -85.7907257, 85.6959076
36: -62.0068016, 24.4158897, -61.8691444, 24.3210697, -86.3278732, 86.2850342
37: -87.1807327, 19.8624191, -86.9854279, 19.6417179, -106.8224487, 106.8478470
38: -69.9994659, 29.2078896, -69.8376923, 28.9704552, -98.9699249, 99.0455780
39: -80.5025558, 30.6326561, -80.2578888, 30.3770256, -110.8795776, 110.8905487
40: -62.5998535, 25.6653023, -62.4100380, 25.2946129, -87.8944702, 88.0753403
41: -55.0261650, 32.8906555, -54.8452034, 32.6787643, -87.7049255, 87.7358551
42: -36.2442932, 26.0598583, -36.1699066, 25.9531975, -62.1974907, 62.2297668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=226, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2263029
time: 117.33 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9909015, upper bound: 44.1846071
time: 1663.03 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -56.7090187, 43.5120316, -56.5305176, 43.3131523, -100.0221710, 100.0425491
1: -25.4061165, 37.7992020, -25.2724419, 37.6065979, -63.0127144, 63.0716438
2: -21.9793396, 37.2488823, -21.8189774, 37.1047974, -59.0841370, 59.0678596
3: -24.6236496, 39.9080582, -24.5128880, 39.8017159, -64.4253693, 64.4209442
4: -28.6711979, 43.7793884, -28.4486160, 43.5001984, -72.1713943, 72.2280045
5: -24.8199043, 39.8242340, -24.6606102, 39.6409035, -64.4608078, 64.4848480
6: -54.3148575, 31.9029293, -54.2594719, 31.7573853, -86.0722427, 86.1623993
7: -30.5713253, 39.5963821, -30.3674774, 39.3495293, -69.9208527, 69.9638596
8: -36.7296371, 53.6428680, -36.5696220, 53.4287643, -90.1584015, 90.2124939
9: -29.1572838, 39.0658646, -29.0801735, 38.9553909, -68.1126709, 68.1460419
10: -49.6490288, 43.9724045, -49.4628639, 43.7759285, -93.4249573, 93.4352722
11: -49.1733475, 22.1397133, -48.9230652, 21.9897633, -71.1631088, 71.0627747
12: -55.3236694, 25.3849335, -55.0427895, 25.1034050, -79.3083649, 79.3073730
13: -50.7016373, 43.8559685, -50.5725822, 43.7669525, -94.4685898, 94.4285507
14: -87.5911102, 31.3887272, -87.1889648, 31.0937138, -118.6848221, 118.5776901
15: -35.9002533, 36.0808296, -35.7330894, 36.0141373, -71.9143906, 71.8139191
16: -45.9883194, 33.9729233, -45.8812218, 33.7269363, -79.7152557, 79.8541412
17: -84.9260712, 23.6288605, -84.5224304, 23.4049950, -108.3310699, 108.1512909
18: -49.1016273, 31.4285583, -49.0682869, 31.3661499, -80.4677734, 80.4968414
19: -39.0137482, 18.6530342, -38.8821411, 18.6013145, -57.6150627, 57.5351753
20: -36.9852867, 23.4739037, -36.8438797, 23.3730221, -60.3583069, 60.3177834
21: -48.2239876, 22.2377968, -48.0201836, 22.1372166, -70.3612061, 70.2579803
22: -50.0124130, 22.1990185, -49.6636810, 22.0638733, -72.0762863, 71.8627014
23: -39.1494217, 24.1386986, -38.9443817, 24.0345039, -63.1839256, 63.0830803
24: -46.3715858, 24.1039734, -46.1855812, 24.0364914, -70.4080811, 70.2895508
25: -41.2684250, 24.8866844, -41.0177803, 24.7418594, -66.0102844, 65.9044647
26: -57.0455017, 33.7465591, -56.7799911, 33.5308762, -90.5763779, 90.5265503
27: -45.3719978, 28.7977180, -45.2807426, 28.7320728, -74.1040726, 74.0784607
28: -38.9973373, 26.8443909, -38.8276672, 26.7497082, -65.7470474, 65.6720581
29: -51.6606522, 20.7218285, -51.2588654, 20.5654659, -72.2261200, 71.9806976
30: -49.3442726, 26.3291588, -49.1040649, 26.2030449, -75.5473175, 75.4332275
31: -51.1490669, 27.9988976, -50.9980316, 27.9092655, -79.0583344, 78.9969330
32: -52.4053421, 24.7156944, -52.2850189, 24.6476402, -77.0529785, 77.0007172
33: -72.4241714, 33.8540039, -72.2713013, 33.7273102, -105.8414612, 105.8256302
34: -65.5431442, 17.1829605, -65.4530258, 17.1111317, -81.8000565, 81.7871857
35: -63.8139114, 23.5792561, -63.6985703, 23.5209866, -85.7947693, 85.7359390
36: -62.0075264, 24.4180088, -61.9166908, 24.3506870, -86.3582153, 86.3347015
37: -87.1832962, 19.8589249, -87.1017456, 19.6732330, -106.8565292, 106.9606705
38: -69.9997787, 29.2146912, -69.9304504, 29.0378666, -99.0376434, 99.1451416
39: -80.5034027, 30.6425724, -80.3464355, 30.4308968, -110.9342957, 110.9890060
40: -62.6016045, 25.6863022, -62.5554848, 25.4063263, -88.0079346, 88.2417908
41: -55.0280457, 32.8980713, -54.9322891, 32.7299232, -87.7579651, 87.8303604
42: -36.2376938, 26.0608635, -36.1771469, 25.9638309, -62.2015228, 62.2380104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=226, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2263029
time: 91.49 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2308690
time: 91.58 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -56.7069016, 43.5079575, -56.5734901, 43.3627930, -100.0696945, 100.0814514
1: -25.4042320, 37.7924843, -25.3071308, 37.6605682, -63.0648003, 63.0996170
2: -21.9779205, 37.2462883, -21.8747768, 37.1490250, -59.1269455, 59.1210632
3: -24.6279335, 39.9069405, -24.5562267, 39.8429947, -64.4709320, 64.4631653
4: -28.6696434, 43.7738533, -28.5269852, 43.5895767, -72.2592163, 72.3008423
5: -24.8198071, 39.8177261, -24.7309856, 39.6984329, -64.5182419, 64.5487137
6: -54.3136902, 31.8963623, -54.2707329, 31.7915573, -86.1052475, 86.1670990
7: -30.5699444, 39.5887299, -30.4467735, 39.4311142, -70.0010605, 70.0355072
8: -36.7288704, 53.6368942, -36.6219711, 53.4830589, -90.2119293, 90.2588654
9: -29.1597939, 39.0626831, -29.1068192, 38.9805756, -68.1403656, 68.1695023
10: -49.6447525, 43.9708023, -49.5482521, 43.8821945, -93.5269470, 93.5190582
11: -49.1652603, 22.1388226, -48.9953346, 22.0451279, -71.2103882, 71.1341553
12: -55.3196716, 25.3854122, -55.1649551, 25.2465401, -79.4342041, 79.4223022
13: -50.7095413, 43.8541031, -50.6286850, 43.8009148, -94.5104523, 94.4827881
14: -87.5806351, 31.3878632, -87.3799896, 31.2465248, -118.8271637, 118.7678528
15: -35.9006271, 36.0820923, -35.8026390, 36.0386353, -71.9392624, 71.8847351
16: -45.9853897, 33.9685287, -45.9053917, 33.7999725, -79.7853622, 79.8739166
17: -84.9118958, 23.6275787, -84.6706390, 23.5076847, -108.4195786, 108.2982178
18: -49.0996857, 31.4448433, -49.0834618, 31.3691292, -80.4688110, 80.5283051
19: -39.0085602, 18.6524620, -38.9050751, 18.6014481, -57.6100082, 57.5575371
20: -36.9824791, 23.4731560, -36.8917236, 23.4080887, -60.3905678, 60.3648796
21: -48.2164268, 22.2371330, -48.0637627, 22.1520081, -70.3684387, 70.3008957
22: -49.9990692, 22.1979427, -49.7640953, 22.0945930, -72.0936584, 71.9620361
23: -39.1385040, 24.1385689, -38.9911423, 24.0517540, -63.1902580, 63.1297112
24: -46.3639946, 24.1031227, -46.2406616, 24.0474911, -70.4114838, 70.3437805
25: -41.2574310, 24.8849163, -41.0914040, 24.7767391, -66.0341721, 65.9763184
26: -57.0420227, 33.7465897, -56.8910255, 33.6273041, -90.6693268, 90.6376190
27: -45.3700409, 28.8070488, -45.3168144, 28.7664280, -74.1364670, 74.1238632
28: -38.9900665, 26.8441715, -38.8664207, 26.7604866, -65.7505493, 65.7105942
29: -51.6427460, 20.7207241, -51.3581963, 20.5941525, -72.2369003, 72.0789185
30: -49.3305130, 26.3288403, -49.1563721, 26.2191391, -75.5496521, 75.4852142
31: -51.1442261, 27.9979534, -51.0405846, 27.9369812, -79.0812073, 79.0385361
32: -52.4095917, 24.7153988, -52.3345985, 24.6756248, -77.0852203, 77.0499954
33: -72.4225235, 33.8510742, -72.3412018, 33.7878952, -106.0217056, 105.9399796
34: -65.5406799, 17.1814156, -65.4686584, 17.1222858, -81.9404755, 81.8823090
35: -63.8137474, 23.5779934, -63.7274704, 23.5438976, -85.9253845, 85.8501892
36: -62.0068016, 24.4158897, -61.9391594, 24.3630943, -86.3698959, 86.3550491
37: -87.1807327, 19.8624191, -87.1227875, 19.7412548, -106.9219894, 106.9852066
38: -69.9994659, 29.2078896, -69.9410934, 29.0786133, -99.0780792, 99.1489868
39: -80.5025558, 30.6326561, -80.4091568, 30.5080147, -111.0105743, 111.0418091
40: -62.5998535, 25.6653023, -62.5273857, 25.4498672, -88.0497208, 88.1926880
41: -55.0261650, 32.8906555, -54.9637833, 32.7820854, -87.8082504, 87.8544388
42: -36.2442932, 26.0598583, -36.1917343, 26.0072575, -62.2515488, 62.2515945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=226, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2265181
time: 82.65 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2308983
time: 113.39 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -56.7090187, 43.5120316, -56.6844559, 43.4125519, -100.1215668, 100.1964874
1: -25.4061165, 37.7992020, -25.3765945, 37.7048645, -63.1109810, 63.1757965
2: -21.9793396, 37.2488823, -21.9400845, 37.1778030, -59.1571426, 59.1889648
3: -24.6236496, 39.9080582, -24.5678978, 39.8593750, -64.4830246, 64.4759521
4: -28.6711979, 43.7793884, -28.6321392, 43.6479759, -72.3191757, 72.4115295
5: -24.8199043, 39.8242340, -24.7638798, 39.7505608, -64.5704651, 64.5881119
6: -54.3148575, 31.9029293, -54.3250389, 31.8422890, -86.1571503, 86.2279663
7: -30.5713253, 39.5963821, -30.5144653, 39.4840240, -70.0553513, 70.1108475
8: -36.7296371, 53.6428680, -36.6822128, 53.5392838, -90.2689209, 90.3250809
9: -29.1572838, 39.0658646, -29.1367359, 39.0139503, -68.1712341, 68.2025986
10: -49.6490288, 43.9724045, -49.5968094, 43.9483490, -93.5973816, 93.5692139
11: -49.1733475, 22.1397133, -49.0679703, 22.1206913, -71.2940369, 71.2076874
12: -55.3236694, 25.3849335, -55.2095490, 25.2795258, -79.4741058, 79.4611969
13: -50.7016373, 43.8559685, -50.6417847, 43.8375893, -94.5392303, 94.4977570
14: -87.5911102, 31.3887272, -87.4670105, 31.3653469, -118.9564590, 118.8557358
15: -35.9002533, 36.0808296, -35.8294449, 36.0616608, -71.9619141, 71.9102783
16: -45.9883194, 33.9729233, -45.9886703, 33.8500137, -79.8383331, 79.9615936
17: -84.9260712, 23.6288605, -84.7752991, 23.5877075, -108.5137787, 108.4041595
18: -49.1016273, 31.4285583, -49.1072617, 31.3679924, -80.4696198, 80.5358200
19: -39.0137482, 18.6530342, -38.9551239, 18.6326752, -57.6464233, 57.6081581
20: -36.9852867, 23.4739037, -36.9255371, 23.4566936, -60.4419785, 60.3994408
21: -48.2239876, 22.2377968, -48.1301460, 22.2168198, -70.4408112, 70.3679428
22: -50.0124130, 22.1990185, -49.8552246, 22.1880493, -72.2004623, 72.0542450
23: -39.1494217, 24.1386986, -39.0664864, 24.1442909, -63.2937126, 63.2051849
24: -46.3715858, 24.1039734, -46.3009186, 24.1181221, -70.4897079, 70.4048920
25: -41.2684250, 24.8866844, -41.1631241, 24.8917732, -66.1602020, 66.0498047
26: -57.0455017, 33.7465591, -56.9366074, 33.6673584, -90.7128601, 90.6831665
27: -45.3719978, 28.7977180, -45.3505554, 28.7667179, -74.1387177, 74.1482697
28: -38.9973373, 26.8443909, -38.9265213, 26.8497028, -65.8470383, 65.7709122
29: -51.6606522, 20.7218285, -51.4788170, 20.7255783, -72.3862305, 72.2006454
30: -49.3442726, 26.3291588, -49.2388802, 26.3577728, -75.7020416, 75.5680389
31: -51.1490669, 27.9988976, -51.0865707, 27.9772644, -79.1263275, 79.0854645
32: -52.4053421, 24.7156944, -52.3593788, 24.6921692, -77.0975113, 77.0750732
33: -72.4241714, 33.8540039, -72.3952408, 33.8213768, -106.0149536, 106.0667038
34: -65.5431442, 17.1829605, -65.5102615, 17.1706867, -81.9233170, 81.9452667
35: -63.8139114, 23.5792561, -63.7622223, 23.5574188, -85.9098511, 85.8922806
36: -62.0075264, 24.4180088, -61.9914322, 24.3929825, -86.4005127, 86.4094391
37: -87.1832962, 19.8589249, -87.2544098, 19.7848148, -106.9681091, 107.1133347
38: -69.9997787, 29.2146912, -70.0375824, 29.1465702, -99.1463470, 99.2522736
39: -80.5034027, 30.6425724, -80.5015793, 30.5665817, -111.0699844, 111.1441498
40: -62.6016045, 25.6863022, -62.6833572, 25.5674629, -88.1690674, 88.3696594
41: -55.0280457, 32.8980713, -55.0542336, 32.8337784, -87.8618240, 87.9523010
42: -36.2376938, 26.0608635, -36.2001190, 26.0193539, -62.2570496, 62.2609825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=228, inp2_unstable=226, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=489, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0558930, upper bound: 44.1203090
time: 80.86 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0313899, upper bound: 44.1417762
time: 88.57 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -56.7261810, 43.5667267, -56.5728912, 43.4682770, -100.1944580, 100.1396179
1: -25.4146233, 37.8481140, -25.3106041, 37.7491302, -63.1637535, 63.1587181
2: -21.9878807, 37.2870941, -21.8665829, 37.2153435, -59.2032242, 59.1536789
3: -24.6365662, 39.9320984, -24.5811958, 39.8737068, -64.5102692, 64.5132904
4: -28.6819763, 43.8510666, -28.5033894, 43.7074814, -72.3894577, 72.3544540
5: -24.8284073, 39.8650436, -24.7248363, 39.7551880, -64.5835953, 64.5898819
6: -54.3239365, 31.9218349, -54.2574234, 31.8334408, -86.1573792, 86.1792603
7: -30.5830002, 39.6546021, -30.4357243, 39.5187988, -70.1017990, 70.0903244
8: -36.7375145, 53.7005386, -36.6252747, 53.5887794, -90.3262939, 90.3258133
9: -29.1692905, 39.0828934, -29.1123333, 39.0239029, -68.1931915, 68.1952286
10: -49.6711693, 43.9869232, -49.5368690, 43.8132477, -93.4844208, 93.5237885
11: -49.2300034, 22.1481342, -49.0849648, 22.0164337, -71.2464371, 71.2331009
12: -55.3852463, 25.3946457, -55.2167854, 25.2185249, -79.4836655, 79.4954681
13: -50.7213821, 43.8688736, -50.6504593, 43.7980881, -94.5194702, 94.5193329
14: -87.6552658, 31.3947296, -87.3757248, 31.1235161, -118.7787781, 118.7704544
15: -35.9208069, 36.0936661, -35.8187866, 36.0449409, -71.9657440, 71.9124527
16: -46.0053177, 34.0045853, -45.8969116, 33.8759499, -79.8812714, 79.9014969
17: -85.0072556, 23.6380692, -84.7534027, 23.4552441, -108.4625015, 108.3914719
18: -49.1181335, 31.4397640, -49.0782280, 31.4440193, -80.5621490, 80.5179901
19: -39.0444946, 18.6569519, -38.9709435, 18.6252766, -57.6697693, 57.6278954
20: -37.0126991, 23.4788189, -36.9313736, 23.3953457, -60.4080429, 60.4101944
21: -48.2729263, 22.2438297, -48.1619606, 22.1645851, -70.4375153, 70.4057922
22: -50.0950737, 22.2046642, -49.9006691, 22.0802155, -72.1752930, 72.1053314
23: -39.1937714, 24.1445808, -39.0715714, 24.0346985, -63.2284698, 63.2161522
24: -46.4078598, 24.1088161, -46.2917404, 24.0273418, -70.4352036, 70.4005585
25: -41.3209534, 24.8949184, -41.1748734, 24.7451935, -66.0661469, 66.0697937
26: -57.0989380, 33.7530136, -56.9410896, 33.6158066, -90.7147446, 90.6941071
27: -45.3869171, 28.8061600, -45.3165321, 28.7636642, -74.1505814, 74.1226959
28: -39.0338364, 26.8494759, -38.9374084, 26.7494354, -65.7832718, 65.7868805
29: -51.7563705, 20.7270565, -51.5334663, 20.5669365, -72.3233032, 72.2605209
30: -49.3985786, 26.3384037, -49.2681961, 26.1879444, -75.5865250, 75.6065979
31: -51.1800880, 28.0046825, -51.0909615, 27.9368210, -79.1169128, 79.0956421
32: -52.4325218, 24.7225780, -52.3559647, 24.6777039, -77.1102295, 77.0785446
33: -72.4320679, 33.8678513, -72.3084106, 33.7734756, -106.0041046, 105.9419861
34: -65.5592270, 17.1944389, -65.4998398, 17.1322556, -81.9205933, 81.8786240
35: -63.8316727, 23.5893116, -63.7667542, 23.5524540, -85.8539581, 85.8129730
36: -62.0250397, 24.4282570, -61.9523163, 24.3857403, -86.4107819, 86.3805695
37: -87.2017822, 19.8828888, -87.0560303, 19.7718983, -106.9736786, 106.9389191
38: -70.0148621, 29.2445507, -69.9058914, 29.1272697, -99.1421356, 99.1504440
39: -80.5200348, 30.6800499, -80.3650131, 30.5475368, -111.0675735, 111.0450592
40: -62.6156960, 25.7342529, -62.4844589, 25.5561485, -88.1718445, 88.2187119
41: -55.0380630, 32.9176598, -54.9134903, 32.8048401, -87.8429031, 87.8311462
42: -36.2526360, 26.0751705, -36.2292023, 26.0154648, -62.2681007, 62.3043747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 615

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2380842
time: 96.44 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2380842
time: 84.55 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -56.7261810, 43.5667267, -56.7261810, 43.5667267, -100.2929077, 100.2929077
1: -25.4146233, 37.8481140, -25.4146233, 37.8481140, -63.2627373, 63.2627373
2: -21.9878807, 37.2870941, -21.9878807, 37.2870941, -59.2749748, 59.2749748
3: -24.6365662, 39.9320984, -24.6365662, 39.9320984, -64.5686646, 64.5686646
4: -28.6819763, 43.8510666, -28.6819763, 43.8510666, -72.5330429, 72.5330429
5: -24.8284073, 39.8650436, -24.8284073, 39.8650436, -64.6934509, 64.6934509
6: -54.3239365, 31.9218349, -54.3239365, 31.9218349, -86.2457733, 86.2457733
7: -30.5830002, 39.6546021, -30.5830002, 39.6546021, -70.2376022, 70.2376022
8: -36.7375145, 53.7005386, -36.7375145, 53.7005386, -90.4380493, 90.4380493
9: -29.1692905, 39.0828934, -29.1692905, 39.0828934, -68.2521820, 68.2521820
10: -49.6711693, 43.9869232, -49.6711693, 43.9869232, -93.6580963, 93.6580963
11: -49.2300034, 22.1481342, -49.2300034, 22.1481342, -71.3781357, 71.3781357
12: -55.3852463, 25.3946457, -55.3852463, 25.3946457, -79.6511688, 79.6511612
13: -50.7213821, 43.8688736, -50.7213821, 43.8688736, -94.5902557, 94.5902557
14: -87.6552658, 31.3947296, -87.6552658, 31.3947296, -119.0499954, 119.0499954
15: -35.9208069, 36.0936661, -35.9208069, 36.0936661, -72.0144730, 72.0144730
16: -46.0053177, 34.0045853, -46.0053177, 34.0045853, -80.0099030, 80.0099030
17: -85.0072556, 23.6380692, -85.0072556, 23.6380692, -108.6453247, 108.6453247
18: -49.1181335, 31.4397640, -49.1181335, 31.4397640, -80.5578995, 80.5578995
19: -39.0444946, 18.6569519, -39.0444946, 18.6569519, -57.7014465, 57.7014465
20: -37.0126991, 23.4788189, -37.0126991, 23.4788189, -60.4915161, 60.4915161
21: -48.2729263, 22.2438297, -48.2729263, 22.2438297, -70.5167542, 70.5167542
22: -50.0950737, 22.2046642, -50.0950737, 22.2046642, -72.2997360, 72.2997360
23: -39.1937714, 24.1445808, -39.1937714, 24.1445808, -63.3383522, 63.3383522
24: -46.4078598, 24.1088161, -46.4078598, 24.1088161, -70.5166779, 70.5166779
25: -41.3209534, 24.8949184, -41.3209534, 24.8949184, -66.2158737, 66.2158737
26: -57.0989380, 33.7530136, -57.0989380, 33.7530136, -90.8519516, 90.8519516
27: -45.3869171, 28.8061600, -45.3869171, 28.8061600, -74.1930771, 74.1930771
28: -39.0338364, 26.8494759, -39.0338364, 26.8494759, -65.8833160, 65.8833160
29: -51.7563705, 20.7270565, -51.7563705, 20.7270565, -72.4834290, 72.4834290
30: -49.3985786, 26.3384037, -49.3985786, 26.3384037, -75.7369843, 75.7369843
31: -51.1800880, 28.0046825, -51.1800880, 28.0046825, -79.1847687, 79.1847687
32: -52.4325218, 24.7225780, -52.4325218, 24.7225780, -77.1550980, 77.1550980
33: -72.4320679, 33.8678513, -72.4320679, 33.8678513, -106.1801605, 106.1801682
34: -65.5592270, 17.1944389, -65.5592270, 17.1944389, -82.0552673, 82.0552673
35: -63.8316727, 23.5893116, -63.8316727, 23.5893116, -85.9716949, 85.9716949
36: -62.0250397, 24.4282570, -62.0250397, 24.4282570, -86.4532928, 86.4532928
37: -87.2017822, 19.8828888, -87.2017822, 19.8828888, -107.0846710, 107.0846710
38: -70.0148621, 29.2445507, -70.0148621, 29.2445507, -99.2594147, 99.2594147
39: -80.5200348, 30.6800499, -80.5200348, 30.6800499, -111.2000885, 111.2000885
40: -62.6156960, 25.7342529, -62.6156960, 25.7342529, -88.3499451, 88.3499451
41: -55.0380630, 32.9176598, -55.0380630, 32.9176598, -87.9557190, 87.9557190
42: -36.2526360, 26.0751705, -36.2526360, 26.0751705, -62.3278046, 62.3278046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0176386, upper bound: 44.2478539
time: 70.39 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0176386, upper bound: 44.0871320
time: 505.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 578.13 seconds
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2263029
IS_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 578.13
Output dim: 4, lower bound: -43.9909015, upper bound: 44.1846071
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2263029
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2308690
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2265181
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2308983
IS_A2_B1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 578.13
Output dim: 4, lower bound: -44.0558930, upper bound: 44.1203090
IS_A2_B1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 578.13
Output dim: 4, lower bound: -44.0313899, upper bound: 44.1417762
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2380842
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -44.0515326, upper bound: 44.2380842
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 578.13
Output dim: 4, lower bound: -44.0176386, upper bound: 44.2478539
IS_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 578.13
Output dim: 4, lower bound: -44.0176386, upper bound: 44.0871320

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -56.5040283, 43.4250031, -56.3635368, 43.2571411, -99.7611694, 99.7885437
1: -25.2520676, 37.6951752, -25.1578541, 37.5518265, -62.8038940, 62.8530273
2: -21.7509422, 37.1220322, -21.6820145, 37.0674934, -58.8184357, 58.8040466
3: -24.3688507, 39.7604370, -24.4180679, 39.7722931, -64.1411438, 64.1785049
4: -28.4204006, 43.6489334, -28.2710972, 43.4390106, -71.8594131, 71.9200287
5: -24.5805550, 39.6900940, -24.5524578, 39.5765457, -64.1571045, 64.2425537
6: -54.2240982, 31.7646961, -54.1868820, 31.6696396, -85.8937378, 85.9515762
7: -30.3700695, 39.5159607, -30.2413521, 39.2855949, -69.6556625, 69.7573090
8: -36.5288582, 53.4968452, -36.4482727, 53.3562202, -89.8850784, 89.9451141
9: -29.0495987, 38.9803200, -29.0192089, 38.9026794, -67.9522781, 67.9995270
10: -49.5037956, 43.7282562, -49.3866806, 43.6385803, -93.1423798, 93.1149368
11: -48.9221191, 21.8303471, -48.8269501, 21.8127975, -70.7349167, 70.6572952
12: -55.1634827, 25.0536766, -54.9816895, 24.9649124, -79.0048828, 78.9113464
13: -50.5106201, 43.7073898, -50.4944992, 43.7036133, -94.2142334, 94.2018890
14: -87.3657990, 31.0337048, -87.0684814, 30.8596287, -118.2254257, 118.1021881
15: -35.6936722, 35.9970779, -35.6420670, 35.9716110, -71.6652832, 71.6391449
16: -45.8312912, 33.7753067, -45.7601471, 33.6204605, -79.4517517, 79.5354538
17: -84.6945190, 23.3501263, -84.3954468, 23.2369404, -107.9314575, 107.7455750
18: -48.9027977, 31.2053108, -49.0281868, 31.2911606, -80.1939545, 80.2334976
19: -38.8328781, 18.4531937, -38.8106995, 18.5046463, -57.3375244, 57.2638931
20: -36.8349380, 23.2895927, -36.7882996, 23.2646427, -60.0995789, 60.0778923
21: -48.0031891, 21.9878654, -47.9307899, 21.9905453, -69.9937363, 69.9186554
22: -49.8770332, 22.0411530, -49.5507202, 21.9226360, -71.7996674, 71.5918732
23: -38.9912567, 23.9340725, -38.8505325, 23.8762264, -62.8674850, 62.7846069
24: -46.2371254, 23.9577827, -46.1052132, 23.9192238, -70.1563492, 70.0629959
25: -41.1467209, 24.7142296, -40.9260674, 24.5741959, -65.7209167, 65.6402969
26: -56.8349648, 33.3839645, -56.7090340, 33.3725433, -90.2075043, 90.0930023
27: -45.2074089, 28.6723576, -45.2240944, 28.6787376, -73.8861465, 73.8964539
28: -38.8391380, 26.6355743, -38.7485046, 26.5932636, -65.4324036, 65.3840790
29: -51.4996567, 20.5210018, -51.1193352, 20.3702793, -71.8699341, 71.6403351
30: -49.1654053, 26.0713596, -49.0077362, 25.9871063, -75.1525116, 75.0790939
31: -50.9412842, 27.7807922, -50.9263687, 27.7983093, -78.7395935, 78.7071609
32: -52.3078232, 24.5960159, -52.2430992, 24.5959606, -76.9037857, 76.8391113
33: -72.2133026, 33.7023163, -72.1533356, 33.6701279, -105.5555191, 105.4824829
34: -65.4468689, 17.0707302, -65.3874359, 17.0355473, -81.6279831, 81.5683441
35: -63.7224617, 23.5039024, -63.6390305, 23.4921970, -85.6664505, 85.5774384
36: -61.9306946, 24.3426514, -61.8504562, 24.3025646, -86.2332611, 86.1931076
37: -87.0700302, 19.7171478, -86.9619675, 19.5961094, -106.6661377, 106.6791153
38: -69.8721390, 29.1254997, -69.8040009, 28.9513359, -98.8234711, 98.9295044
39: -80.3354492, 30.5283012, -80.2072144, 30.3603592, -110.6958084, 110.7355194
40: -62.4936028, 25.5625267, -62.3856316, 25.2714119, -87.7650146, 87.9481583
41: -54.9364395, 32.7661133, -54.8220329, 32.6454163, -87.5818558, 87.5881500
42: -36.1716843, 25.9141655, -36.1526985, 25.9130630, -62.0847473, 62.0668640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=226, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1550352
time: 94.25 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9888773, upper bound: 44.2243924
time: 78.37 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -56.5060883, 43.4291077, -56.4711266, 43.2973709, -99.8034592, 99.9002380
1: -25.2539368, 37.7019501, -25.2259941, 37.5959587, -62.8498955, 62.9279442
2: -21.7523270, 37.1246033, -21.7454643, 37.0931435, -58.8454704, 58.8700676
3: -24.3645363, 39.7615204, -24.4286995, 39.7875786, -64.1521149, 64.1902161
4: -28.4219017, 43.6544189, -28.3682194, 43.4884491, -71.9103546, 72.0226364
5: -24.5806713, 39.6966171, -24.5840263, 39.6271667, -64.2078400, 64.2806396
6: -54.2252502, 31.7709808, -54.2391434, 31.7186642, -85.9439163, 86.0101242
7: -30.3714561, 39.5236053, -30.3073368, 39.3383255, -69.7097778, 69.8309402
8: -36.5296135, 53.5028152, -36.5064278, 53.4120026, -89.9416199, 90.0092468
9: -29.0469017, 38.9835281, -29.0473709, 38.9341354, -67.9810333, 68.0308990
10: -49.5081253, 43.7297821, -49.4345245, 43.7023201, -93.2104492, 93.1643066
11: -48.9301376, 21.8312206, -48.8981934, 21.8868332, -70.8169708, 70.7294159
12: -55.1674881, 25.0532036, -55.0258827, 24.9957352, -79.0426025, 78.9498138
13: -50.5026131, 43.7092133, -50.5063324, 43.7379303, -94.2405396, 94.2155457
14: -87.3762360, 31.0345135, -87.1549530, 30.9752502, -118.3514862, 118.1894684
15: -35.6932487, 35.9957733, -35.6670227, 35.9945831, -71.6878357, 71.6627960
16: -45.8341446, 33.7794952, -45.8423462, 33.6697464, -79.5038910, 79.6218414
17: -84.7087097, 23.3513508, -84.4998627, 23.3137512, -108.0224609, 107.8512115
18: -48.9046860, 31.1901016, -49.0489044, 31.2827148, -80.1874008, 80.2390060
19: -38.8380318, 18.4537430, -38.8596191, 18.5351410, -57.3731728, 57.3133621
20: -36.8377151, 23.2903366, -36.8199654, 23.3122635, -60.1499786, 60.1103020
21: -48.0107231, 21.9885406, -47.9948921, 22.0538826, -70.0646057, 69.9834290
22: -49.8903809, 22.0421791, -49.6410599, 22.0146446, -71.9050293, 71.6832428
23: -39.0021515, 23.9342003, -38.9249954, 23.9675694, -62.9697189, 62.8591957
24: -46.2446861, 23.9586411, -46.1640930, 23.9888382, -70.2335205, 70.1227341
25: -41.1577148, 24.7159519, -40.9974060, 24.6862164, -65.8439331, 65.7133560
26: -56.8384476, 33.3839455, -56.7537766, 33.4117737, -90.2502213, 90.1377258
27: -45.2093163, 28.6631088, -45.2553177, 28.6785393, -73.8878555, 73.9184265
28: -38.8463821, 26.6358128, -38.8053284, 26.6811581, -65.5275421, 65.4411392
29: -51.5175209, 20.5220547, -51.2395134, 20.4998646, -72.0173874, 71.7615662
30: -49.1791573, 26.0716782, -49.0826263, 26.1193562, -75.2985153, 75.1543045
31: -50.9461479, 27.7816944, -50.9715195, 27.8372498, -78.7834015, 78.7532120
32: -52.3024788, 24.5962944, -52.2619667, 24.6118240, -76.9143066, 76.8582611
33: -72.2148590, 33.7052383, -72.2044754, 33.7029877, -105.5786133, 105.6007614
34: -65.4491882, 17.0722580, -65.4266052, 17.0835247, -81.6517105, 81.6264343
35: -63.7225533, 23.5051289, -63.6719704, 23.5054569, -85.6714935, 85.6177902
36: -61.9313965, 24.3447247, -61.8979568, 24.3321915, -86.2635880, 86.2426834
37: -87.0725250, 19.7136574, -87.0782089, 19.6276073, -106.7001343, 106.7918701
38: -69.8723602, 29.1322918, -69.8966827, 29.0187397, -98.8910980, 99.0289764
39: -80.3363342, 30.5382309, -80.2957230, 30.4142685, -110.7506027, 110.8339539
40: -62.4952469, 25.5835228, -62.5308762, 25.3824635, -87.8777084, 88.1143951
41: -54.9382858, 32.7734947, -54.9090996, 32.6958923, -87.6341782, 87.6825943
42: -36.1650620, 25.9148941, -36.1599655, 25.9236469, -62.0887070, 62.0748596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=226, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1550352
time: 81.22 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2243924
time: 155.52 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -56.6961632, 43.5073471, -56.5263824, 43.3116570, -100.0078201, 100.0337296
1: -25.3959427, 37.7948227, -25.2693081, 37.6052246, -63.0011673, 63.0641327
2: -21.9643173, 37.2448311, -21.8145180, 37.1035728, -59.0678902, 59.0593491
3: -24.6067219, 39.9030762, -24.5078735, 39.8001442, -64.4068680, 64.4109497
4: -28.6543617, 43.7758255, -28.4436798, 43.4990845, -72.1534424, 72.2195053
5: -24.8044014, 39.8200378, -24.6560440, 39.6395645, -64.4439697, 64.4760818
6: -54.3091278, 31.8650513, -54.2577438, 31.7456093, -86.0547333, 86.1227951
7: -30.5575275, 39.5897598, -30.3633194, 39.3474045, -69.9049301, 69.9530792
8: -36.7163696, 53.6365280, -36.5656738, 53.4267464, -90.1431122, 90.2022018
9: -29.1463127, 39.0578499, -29.0765686, 38.9530754, -68.0993881, 68.1344147
10: -49.6419678, 43.9489365, -49.4606857, 43.7690468, -93.4110107, 93.4096222
11: -49.1644859, 22.1179371, -48.9202538, 21.9830990, -71.1475830, 71.0381927
12: -55.3169785, 25.3618774, -55.0406189, 25.0966759, -79.2947769, 79.2597961
13: -50.6503029, 43.8478889, -50.5565796, 43.7644882, -94.4147949, 94.4044647
14: -87.5799942, 31.3648949, -87.1852875, 31.0868282, -118.6668243, 118.5501862
15: -35.8510208, 36.0744476, -35.7154922, 36.0121956, -71.8632202, 71.7899399
16: -45.9768143, 33.9098015, -45.8776321, 33.7067413, -79.6835556, 79.7874298
17: -84.9171295, 23.6094856, -84.5195465, 23.3991928, -108.3163223, 108.1290283
18: -49.0942650, 31.4058495, -49.0659981, 31.3601017, -80.4543686, 80.4718475
19: -39.0073624, 18.6400375, -38.8802567, 18.5973911, -57.6047516, 57.5202942
20: -36.9788742, 23.4615364, -36.8418617, 23.3692322, -60.3481064, 60.3033981
21: -48.2170639, 22.2214336, -48.0180855, 22.1322441, -70.3493042, 70.2395172
22: -50.0023918, 22.1874657, -49.6604576, 22.0602875, -72.0626831, 71.8479233
23: -39.1439056, 24.1250381, -38.9426498, 24.0303841, -63.1742897, 63.0676880
24: -46.3642159, 24.0941734, -46.1832619, 24.0333481, -70.3975677, 70.2774353
25: -41.2603989, 24.8741169, -41.0153198, 24.7379017, -65.9982986, 65.8894348
26: -57.0341415, 33.7223587, -56.7763443, 33.5236282, -90.5577698, 90.4987030
27: -45.3650360, 28.7757645, -45.2785721, 28.7269897, -74.0920258, 74.0543365
28: -38.9916153, 26.8303680, -38.8258972, 26.7454624, -65.7370758, 65.6562653
29: -51.6529198, 20.7081985, -51.2563362, 20.5614128, -72.2143326, 71.9645386
30: -49.3367157, 26.3099327, -49.1016502, 26.1970215, -75.5337372, 75.4115829
31: -51.1424103, 27.9847717, -50.9960709, 27.9047432, -79.0471497, 78.9808426
32: -52.3989716, 24.6984329, -52.2829704, 24.6419678, -77.0409393, 76.9813995
33: -72.4084625, 33.8471336, -72.2664490, 33.7250748, -105.7949219, 105.8136978
34: -65.5317154, 17.1730499, -65.4497986, 17.1081676, -81.7746887, 81.7634277
35: -63.7909775, 23.5751667, -63.6913071, 23.5197620, -85.7704773, 85.7259293
36: -61.9961472, 24.4115334, -61.9131126, 24.3485966, -86.3447418, 86.3246460
37: -87.1746063, 19.8288765, -87.0989151, 19.6646824, -106.8392868, 106.9277954
38: -69.9856110, 29.2021027, -69.9259415, 29.0342236, -99.0198364, 99.1280441
39: -80.4886475, 30.6375904, -80.3418579, 30.4293747, -110.9180222, 110.9794464
40: -62.5944862, 25.6588173, -62.5531349, 25.3968010, -87.9912872, 88.2119522
41: -55.0218811, 32.8693275, -54.9304047, 32.7204895, -87.7423706, 87.7997284
42: -36.2331581, 26.0354519, -36.1757774, 25.9554520, -62.1886101, 62.2112274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=226, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0313899, upper bound: 44.1199662
time: 83.68 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0313899, upper bound: 44.2308690
time: 60.01 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -56.5040283, 43.4250031, -56.5140877, 43.3470612, -99.8510895, 99.9390869
1: -25.2520676, 37.6951752, -25.2607765, 37.6500015, -62.9020691, 62.9559517
2: -21.7509422, 37.1220322, -21.8013077, 37.1374054, -58.8883476, 58.9233398
3: -24.3688507, 39.7604370, -24.4719887, 39.8288956, -64.1977463, 64.2324219
4: -28.4204006, 43.6489334, -28.4466476, 43.5778923, -71.9982910, 72.0955811
5: -24.5805550, 39.6900940, -24.6544323, 39.6846924, -64.2652435, 64.3445282
6: -54.2240982, 31.7646961, -54.2504120, 31.7530193, -85.9771194, 86.0151062
7: -30.3700695, 39.5159607, -30.3866882, 39.4200668, -69.7901382, 69.9026489
8: -36.5288582, 53.4968452, -36.5588913, 53.4663849, -89.9952393, 90.0557404
9: -29.0495987, 38.9803200, -29.0740795, 38.9595261, -68.0091248, 68.0543976
10: -49.5037956, 43.7282562, -49.5199966, 43.8087196, -93.3125153, 93.2482529
11: -48.9221191, 21.8303471, -48.9705200, 21.9422264, -70.8643494, 70.8008652
12: -55.1634827, 25.0536766, -55.1481400, 25.1388168, -79.1682510, 79.0642090
13: -50.5106201, 43.7073898, -50.5624619, 43.7719002, -94.2825165, 94.2698517
14: -87.3657990, 31.0337048, -87.3462067, 31.1278057, -118.4936066, 118.3799133
15: -35.6936722, 35.9970779, -35.7387505, 36.0189972, -71.7126694, 71.7358246
16: -45.8312912, 33.7753067, -45.8667145, 33.7433128, -79.5746002, 79.6420212
17: -84.6945190, 23.3501263, -84.6481857, 23.4162788, -108.1107941, 107.9983139
18: -48.9027977, 31.2053108, -49.0642433, 31.2855549, -80.1883545, 80.2695541
19: -38.8328781, 18.4531937, -38.8826141, 18.5353699, -57.3682480, 57.3358078
20: -36.8349380, 23.2895927, -36.8678551, 23.3473282, -60.1822662, 60.1574478
21: -48.0031891, 21.9878654, -48.0385208, 22.0687256, -70.0719147, 70.0263824
22: -49.8770332, 22.0411530, -49.7418747, 22.0453415, -71.9223785, 71.7830276
23: -38.9912567, 23.9340725, -38.9718056, 23.9848442, -62.9761009, 62.9058762
24: -46.2371254, 23.9577827, -46.2192917, 23.9998398, -70.2369690, 70.1770782
25: -41.1467209, 24.7142296, -41.0711136, 24.7210579, -65.8677826, 65.7853394
26: -56.8349648, 33.3839645, -56.8649864, 33.5081482, -90.3431091, 90.2489471
27: -45.2074089, 28.6723576, -45.2914543, 28.7166939, -73.9241028, 73.9638138
28: -38.8391380, 26.6355743, -38.8441391, 26.6920128, -65.5311508, 65.4797134
29: -51.4996567, 20.5210018, -51.3389893, 20.5285034, -72.0281601, 71.8599930
30: -49.1654053, 26.0713596, -49.1349792, 26.1355000, -75.3009033, 75.2063370
31: -50.9412842, 27.7807922, -51.0141296, 27.8650398, -78.8063202, 78.7949219
32: -52.3078232, 24.5960159, -52.3116226, 24.6399574, -76.9477844, 76.9076385
33: -72.2133026, 33.7023163, -72.2742310, 33.7637253, -105.7577362, 105.7150345
34: -65.4468689, 17.0707302, -65.4427185, 17.0949287, -81.7893829, 81.7225647
35: -63.7224617, 23.5039024, -63.7008591, 23.5284519, -85.7971344, 85.7323456
36: -61.9306946, 24.3426514, -61.9201813, 24.3446426, -86.2753372, 86.2628326
37: -87.0700302, 19.7171478, -87.0986481, 19.6972046, -106.7672348, 106.8157959
38: -69.8721390, 29.1254997, -69.9075012, 29.0599880, -98.9321289, 99.0330048
39: -80.3354492, 30.5283012, -80.3582306, 30.4914608, -110.8269119, 110.8865356
40: -62.4936028, 25.5625267, -62.5026588, 25.4262867, -87.9198914, 88.0651855
41: -54.9364395, 32.7661133, -54.9404411, 32.7475739, -87.6840134, 87.7065582
42: -36.1716843, 25.9141655, -36.1745377, 25.9666996, -62.1383820, 62.0887032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=226, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1552406
time: 117.30 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2246076
time: 79.09 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -56.6940689, 43.5032768, -56.5692520, 43.3612480, -100.0553131, 100.0725250
1: -25.3940659, 37.7881088, -25.3038464, 37.6590652, -63.0531311, 63.0919571
2: -21.9628983, 37.2422562, -21.8701839, 37.1476669, -59.1105652, 59.1124420
3: -24.6110191, 39.9019699, -24.5510998, 39.8412933, -64.4523163, 64.4530716
4: -28.6528244, 43.7703094, -28.5218983, 43.5883102, -72.2411346, 72.2922058
5: -24.8042946, 39.8135223, -24.7262974, 39.6970100, -64.5013046, 64.5398178
6: -54.3080063, 31.8585129, -54.2689247, 31.7795944, -86.0876007, 86.1274414
7: -30.5561428, 39.5820999, -30.4424591, 39.4288597, -69.9850006, 70.0245590
8: -36.7155952, 53.6305389, -36.6178665, 53.4809341, -90.1965332, 90.2484055
9: -29.1488228, 39.0546722, -29.1029644, 38.9780884, -68.1269073, 68.1576385
10: -49.6376419, 43.9473801, -49.5457649, 43.8750153, -93.5126572, 93.4931488
11: -49.1564178, 22.1170750, -48.9922943, 22.0383263, -71.1947479, 71.1093674
12: -55.3129921, 25.3622990, -55.1627121, 25.2395782, -79.4204254, 79.3745422
13: -50.6582489, 43.8460503, -50.6123886, 43.7983780, -94.4566269, 94.4584351
14: -87.5695190, 31.3640327, -87.3761444, 31.2394924, -118.8090134, 118.7401733
15: -35.8511925, 36.0756912, -35.7849197, 36.0364532, -71.8876495, 71.8606110
16: -45.9738884, 33.9034843, -45.9013519, 33.7794571, -79.7533417, 79.8048401
17: -84.9029846, 23.6081848, -84.6675797, 23.5017338, -108.4047165, 108.2757645
18: -49.0923309, 31.4221611, -49.0810585, 31.3627396, -80.4550705, 80.5032196
19: -39.0021553, 18.6394749, -38.9029694, 18.5974216, -57.5995789, 57.5424423
20: -36.9760742, 23.4608002, -36.8895950, 23.4042549, -60.3803291, 60.3503952
21: -48.2094955, 22.2207451, -48.0614471, 22.1469154, -70.3564148, 70.2821960
22: -49.9890213, 22.1863976, -49.7606812, 22.0908508, -72.0798721, 71.9470825
23: -39.1329956, 24.1249123, -38.9892426, 24.0475426, -63.1805382, 63.1141548
24: -46.3566246, 24.0932999, -46.2382164, 24.0443134, -70.4009399, 70.3315125
25: -41.2493401, 24.8723087, -41.0887260, 24.7727280, -66.0220642, 65.9610367
26: -57.0306587, 33.7224045, -56.8870811, 33.6197624, -90.6504211, 90.6094818
27: -45.3630829, 28.7851334, -45.3145714, 28.7597656, -74.1228485, 74.0997009
28: -38.9843140, 26.8301220, -38.8645554, 26.7561016, -65.7404175, 65.6946793
29: -51.6350365, 20.7071304, -51.3555298, 20.5900154, -72.2250519, 72.0626602
30: -49.3228951, 26.3096008, -49.1537552, 26.2130089, -75.5359039, 75.4633560
31: -51.1375389, 27.9838276, -51.0384178, 27.9324207, -79.0699615, 79.0222473
32: -52.4031906, 24.6981163, -52.3324966, 24.6697464, -77.0729370, 77.0306091
33: -72.4068604, 33.8441811, -72.3362885, 33.7856026, -105.9757080, 105.9279633
34: -65.5292511, 17.1715031, -65.4651566, 17.1188812, -81.9143066, 81.8543091
35: -63.7908020, 23.5739021, -63.7201767, 23.5425758, -85.9020844, 85.8400421
36: -61.9954071, 24.4094315, -61.9355583, 24.3609562, -86.3563614, 86.3449860
37: -87.1719971, 19.8323536, -87.1197433, 19.7324181, -106.9044189, 106.9520950
38: -69.9853058, 29.1952877, -69.9363785, 29.0744476, -99.0597534, 99.1316681
39: -80.4878235, 30.6276665, -80.4043274, 30.5063839, -110.9942093, 111.0319977
40: -62.5927658, 25.6378098, -62.5249825, 25.4402122, -88.0329742, 88.1627960
41: -55.0199509, 32.8619156, -54.9617538, 32.7730103, -87.7929611, 87.8236694
42: -36.2397423, 26.0344391, -36.1902542, 25.9986801, -62.2384224, 62.2246933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=226, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0313899, upper bound: 44.1203090
time: 91.26 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0313899, upper bound: 44.2308984
time: 80.61 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -56.7212524, 43.5555878, -56.5510406, 43.4192886, -100.1405411, 100.1066284
1: -25.4120731, 37.8388214, -25.2990913, 37.7068367, -63.1189117, 63.1379128
2: -21.9853706, 37.2808075, -21.8557644, 37.1874771, -59.1728477, 59.1365738
3: -24.6328392, 39.9297066, -24.5645638, 39.8639221, -64.4967651, 64.4942703
4: -28.6786404, 43.8366089, -28.4894676, 43.6431923, -72.3218307, 72.3260803
5: -24.8262100, 39.8559303, -24.7153034, 39.7139397, -64.5401459, 64.5712357
6: -54.3215942, 31.9134293, -54.2471771, 31.7959900, -86.1175842, 86.1606064
7: -30.5800457, 39.6441727, -30.4227161, 39.4705505, -70.0505981, 70.0668869
8: -36.7359924, 53.6895218, -36.6187439, 53.5385284, -90.2745209, 90.3082657
9: -29.1666698, 39.0776978, -29.1012135, 39.0005341, -68.1672058, 68.1789093
10: -49.6614304, 43.9837723, -49.4947433, 43.7993851, -93.4608154, 93.4785156
11: -49.2153091, 22.1464500, -49.0172806, 22.0094280, -71.2247391, 71.1637268
12: -55.3777390, 25.3927345, -55.1838303, 25.2102871, -79.4553070, 79.4512100
13: -50.7192116, 43.8649254, -50.6414146, 43.7813950, -94.5006104, 94.5063400
14: -87.6415100, 31.3926201, -87.3121185, 31.1147099, -118.7562180, 118.7047424
15: -35.9170227, 36.0899696, -35.8045502, 36.0271950, -71.9442139, 71.8945160
16: -46.0005684, 33.9938889, -45.8757553, 33.8278008, -79.8283691, 79.8696442
17: -84.9901733, 23.6355991, -84.6741486, 23.4446526, -108.4348297, 108.3097458
18: -49.1130219, 31.4302254, -49.0579681, 31.4029045, -80.5159302, 80.4881897
19: -39.0369568, 18.6561470, -38.9366150, 18.6217117, -57.6586685, 57.5927620
20: -37.0077705, 23.4773788, -36.9095421, 23.3888931, -60.3966637, 60.3869209
21: -48.2621880, 22.2425518, -48.1141167, 22.1589642, -70.4211502, 70.3566666
22: -50.0783844, 22.2027969, -49.8234596, 22.0723476, -72.1507339, 72.0262604
23: -39.1802902, 24.1431274, -39.0091743, 24.0282135, -63.2085037, 63.1523018
24: -46.3974075, 24.1074524, -46.2443352, 24.0212631, -70.4186707, 70.3517914
25: -41.3068085, 24.8916817, -41.1092606, 24.7314072, -66.0382156, 66.0009460
26: -57.0934677, 33.7518845, -56.9169502, 33.6109543, -90.7044220, 90.6688385
27: -45.3833961, 28.8047371, -45.3014030, 28.7577610, -74.1411591, 74.1061401
28: -39.0230179, 26.8480797, -38.8885498, 26.7434807, -65.7664948, 65.7366333
29: -51.7346382, 20.7250862, -51.4325485, 20.5588646, -72.2935028, 72.1576385
30: -49.3810883, 26.3360596, -49.1934280, 26.1777153, -75.5588074, 75.5294876
31: -51.1721954, 28.0033894, -51.0548553, 27.9311199, -79.1033173, 79.0582428
32: -52.4285965, 24.7211304, -52.3417511, 24.6713696, -77.0999680, 77.0628815
33: -72.4284821, 33.8621750, -72.2927246, 33.7475166, -105.9456329, 105.8901138
34: -65.5515747, 17.1920853, -65.4655457, 17.1215725, -81.8863983, 81.8425674
35: -63.8259087, 23.5875301, -63.7418747, 23.5443325, -85.8382797, 85.7904968
36: -62.0224876, 24.4234619, -61.9414406, 24.3639488, -86.3864365, 86.3648987
37: -87.1967010, 19.8668842, -87.0340958, 19.7014008, -106.8981018, 106.9009781
38: -70.0116119, 29.2282715, -69.8920898, 29.0517998, -99.0634155, 99.1203613
39: -80.5162964, 30.6669025, -80.3488388, 30.4897385, -111.0060349, 111.0157394
40: -62.6116180, 25.7096939, -62.4670868, 25.4468956, -88.0585175, 88.1767807
41: -55.0350876, 32.9081535, -54.9002037, 32.7608566, -87.7959442, 87.8083572
42: -36.2494087, 26.0721436, -36.2152100, 26.0037460, -62.2531548, 62.2873535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=226, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=485, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2299835
time: 89.05 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2323576
time: 117.81 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -56.7233810, 43.5596924, -56.6589088, 43.4622345, -100.1856155, 100.2185974
1: -25.4139652, 37.8456001, -25.3674717, 37.7502060, -63.1641693, 63.2130737
2: -21.9868259, 37.2833900, -21.9198914, 37.2119370, -59.1987610, 59.2032814
3: -24.6285706, 39.9308395, -24.5752754, 39.8787460, -64.5073166, 64.5061188
4: -28.6802101, 43.8421288, -28.5880165, 43.6919861, -72.3721924, 72.4301453
5: -24.8263645, 39.8624573, -24.7473526, 39.7644234, -64.5907898, 64.6098099
6: -54.3227539, 31.9199486, -54.2977448, 31.8455925, -86.1683502, 86.2176971
7: -30.5814476, 39.6518326, -30.4892464, 39.5227051, -70.1041565, 70.1410828
8: -36.7368088, 53.6955261, -36.6775131, 53.5928650, -90.3296738, 90.3730392
9: -29.1641331, 39.0809326, -29.1295204, 39.0329475, -68.1970825, 68.2104492
10: -49.6657143, 43.9853630, -49.5423393, 43.8634644, -93.5291748, 93.5277023
11: -49.2234077, 22.1473484, -49.0877914, 22.0838318, -71.3072357, 71.2351379
12: -55.3817673, 25.3922939, -55.2248955, 25.2418041, -79.5023270, 79.4875946
13: -50.7112961, 43.8668137, -50.6528473, 43.8166008, -94.5278931, 94.5196609
14: -87.6520081, 31.3934784, -87.3976440, 31.2315884, -118.8835983, 118.7911224
15: -35.9170113, 36.0887527, -35.8303299, 36.0499687, -71.9669800, 71.9190826
16: -46.0035133, 34.0002022, -45.9591484, 33.8826218, -79.8861389, 79.9593506
17: -85.0043793, 23.6368809, -84.7782822, 23.5229607, -108.5273438, 108.4151611
18: -49.1149597, 31.4139519, -49.0790901, 31.3940334, -80.5089951, 80.4930420
19: -39.0421638, 18.6567173, -38.9855804, 18.6522236, -57.6943893, 57.6422958
20: -37.0105972, 23.4781265, -36.9406013, 23.4366550, -60.4472504, 60.4187279
21: -48.2697754, 22.2432156, -48.1775932, 22.2223816, -70.4921570, 70.4208069
22: -50.0917625, 22.2038803, -49.9132652, 22.1646481, -72.2564087, 72.1171417
23: -39.1912155, 24.1432438, -39.0832024, 24.1197796, -63.3109970, 63.2264481
24: -46.4050064, 24.1083260, -46.3037453, 24.0908718, -70.4958801, 70.4120712
25: -41.3178482, 24.8934441, -41.1804504, 24.8445950, -66.1624451, 66.0738983
26: -57.0969696, 33.7518845, -56.9606094, 33.6501541, -90.7471237, 90.7124939
27: -45.3853455, 28.7954407, -45.3327408, 28.7569885, -74.1423340, 74.1281815
28: -39.0302811, 26.8483467, -38.9452591, 26.8317146, -65.8619995, 65.7936096
29: -51.7525482, 20.7262001, -51.5516319, 20.6888962, -72.4414444, 72.2778320
30: -49.3948860, 26.3363953, -49.2681046, 26.3096161, -75.7044983, 75.6044998
31: -51.1771278, 28.0043411, -51.0997620, 27.9701157, -79.1472473, 79.1041031
32: -52.4243965, 24.7214432, -52.3576317, 24.6872368, -77.1116333, 77.0790710
33: -72.4301071, 33.8651581, -72.3443985, 33.7801285, -105.9677658, 106.0297012
34: -65.5540161, 17.1936722, -65.5056000, 17.1683350, -81.9077148, 81.9112320
35: -63.8261147, 23.5887966, -63.7763977, 23.5574780, -85.8336487, 85.8308334
36: -62.0232468, 24.4255676, -61.9866753, 24.3934917, -86.4167404, 86.4122467
37: -87.1992340, 19.8634529, -87.1521606, 19.7359905, -106.9352264, 107.0156097
38: -70.0119324, 29.2351379, -69.9825592, 29.1191807, -99.1311111, 99.2176971
39: -80.5172272, 30.6767998, -80.4373322, 30.5434990, -111.0607300, 111.1141357
40: -62.6133766, 25.7306995, -62.6175652, 25.5637035, -88.1770782, 88.3482666
41: -55.0369568, 32.9155884, -54.9872055, 32.8119812, -87.8489380, 87.9027939
42: -36.2428207, 26.0731773, -36.2203217, 26.0154362, -62.2582550, 62.2934990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=226, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=487, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 638

## Relational analysis of IS_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0495353, upper bound: 44.1668045
time: 90.57 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0496451, upper bound: 44.2361733
time: 76.01 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -56.6535034, 43.5513763, -56.7033844, 43.5618668, -100.2153702, 100.2547607
1: -25.3469334, 37.8370743, -25.3935165, 37.8446350, -63.1915665, 63.2305908
2: -21.9271183, 37.2772064, -21.9689445, 37.2840080, -59.2111282, 59.2461510
3: -24.5503616, 39.9194870, -24.6096859, 39.9281387, -64.4785004, 64.5291748
4: -28.5973282, 43.8383560, -28.6556206, 43.8470764, -72.4444046, 72.4939728
5: -24.7525883, 39.8521919, -24.8047619, 39.8610039, -64.6135941, 64.6569519
6: -54.2992401, 31.8823395, -54.3161850, 31.9092655, -86.2085037, 86.1985245
7: -30.5146637, 39.6472244, -30.5616798, 39.6522827, -70.1669464, 70.2089081
8: -36.6597977, 53.6875687, -36.7132378, 53.6964684, -90.3562622, 90.4008026
9: -29.0990829, 39.0741692, -29.1473789, 39.0801620, -68.1792450, 68.2215500
10: -49.6342812, 43.9446907, -49.6597214, 43.9733658, -93.6076508, 93.6044159
11: -49.2066193, 22.0765953, -49.2226753, 22.1258411, -71.3324585, 71.2992706
12: -55.3687477, 25.3315163, -55.3800697, 25.3749332, -79.6126099, 79.5789185
13: -50.6496811, 43.8484840, -50.6984711, 43.8624916, -94.5121765, 94.5469513
14: -87.6197815, 31.3250217, -87.6441498, 31.3729744, -118.9927521, 118.9691696
15: -35.8544655, 36.0748672, -35.9000778, 36.0877953, -71.9422607, 71.9749451
16: -45.9558220, 33.9845543, -45.9899063, 33.9977112, -79.9535370, 79.9744568
17: -84.9771729, 23.5586529, -84.9978256, 23.6133194, -108.5904922, 108.5564804
18: -49.1031914, 31.3652592, -49.1134186, 31.4164543, -80.5196457, 80.4786758
19: -39.0283051, 18.5905571, -39.0394135, 18.6362514, -57.6645584, 57.6299706
20: -36.9943466, 23.4253826, -37.0069580, 23.4621658, -60.4565125, 60.4323425
21: -48.2510834, 22.1849251, -48.2660713, 22.2254696, -70.4765549, 70.4509964
22: -50.0795479, 22.1531048, -50.0901833, 22.1885910, -72.2681427, 72.2432861
23: -39.1774139, 24.0591888, -39.1886368, 24.1179810, -63.2953949, 63.2478256
24: -46.3890114, 24.0275192, -46.4019127, 24.0834465, -70.4724579, 70.4294281
25: -41.3027000, 24.8091049, -41.3152428, 24.8681431, -66.1708450, 66.1243439
26: -57.0799408, 33.6662979, -57.0929642, 33.7260056, -90.8059464, 90.7592621
27: -45.3692017, 28.7558613, -45.3813248, 28.7903004, -74.1595001, 74.1371841
28: -39.0190239, 26.7678566, -39.0292015, 26.8240395, -65.8430634, 65.7970581
29: -51.7403984, 20.6722183, -51.7513504, 20.7099457, -72.4503479, 72.4235687
30: -49.3754578, 26.2559967, -49.3912964, 26.3127155, -75.6881714, 75.6472931
31: -51.1611862, 27.9229965, -51.1741676, 27.9792004, -79.1403885, 79.0971680
32: -52.4095840, 24.6943531, -52.4253349, 24.7137413, -77.1233215, 77.1196899
33: -72.4063568, 33.8439789, -72.4239578, 33.8603897, -106.1444550, 106.1461029
34: -65.5417404, 17.1364479, -65.5537186, 17.1763725, -82.0192566, 81.9911499
35: -63.8168526, 23.5603485, -63.8270073, 23.5803280, -85.9364166, 85.9243546
36: -62.0137787, 24.4042072, -62.0214996, 24.4207706, -86.4345474, 86.4257050
37: -87.1735382, 19.8258247, -87.1928940, 19.8651142, -107.0386505, 107.0187225
38: -69.9923706, 29.2145138, -70.0077972, 29.2352238, -99.2275925, 99.2223129
39: -80.4845123, 30.6662941, -80.5088806, 30.6757698, -111.1602783, 111.1751709
40: -62.5868912, 25.7048187, -62.6066170, 25.7247753, -88.3116684, 88.3114319
41: -55.0166779, 32.8817978, -55.0313110, 32.9064445, -87.9231262, 87.9131088
42: -36.2317390, 26.0373936, -36.2460823, 26.0633755, -62.2951126, 62.2834778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=226, inp2_unstable=227, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=488, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 614
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1413
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 1464
type: B, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -44.0111776, upper bound: 44.1950294
time: 86.97 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -44.0022306, upper bound: 44.2363184
time: 92.29 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 181.72 seconds
IS_A2_B1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1550352
IS_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9888773, upper bound: 44.2243924
IS_A2_B1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1550352
IS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2243924
IS_A2_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0313899, upper bound: 44.1199662
IS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0313899, upper bound: 44.2308690
IS_A2_B1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1552406
IS_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2246076
IS_A2_B1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0313899, upper bound: 44.1203090
IS_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0313899, upper bound: 44.2308984
IS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2299835
IS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2323576
IS_A2_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0495353, upper bound: 44.1668045
IS_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0496451, upper bound: 44.2361733
IS_A2_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0111776, upper bound: 44.1950294
IS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 181.72
Output dim: 4, lower bound: -44.0022306, upper bound: 44.2363184

## BFS IS instance: IS_A2_B1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -56.5019455, 43.4236565, -56.3734932, 43.2939148, -99.7958603, 99.7971497
1: -25.2503109, 37.6943817, -25.1567631, 37.5934830, -62.8437958, 62.8511429
2: -21.7495365, 37.1213379, -21.6830788, 37.1103134, -58.8598480, 58.8044167
3: -24.3667355, 39.7597122, -24.4189186, 39.8436050, -64.2103424, 64.1786346
4: -28.4182243, 43.6482620, -28.2696266, 43.5062218, -71.9244461, 71.9178925
5: -24.5786228, 39.6894150, -24.5534496, 39.6389580, -64.2175827, 64.2428665
6: -54.2227631, 31.7633457, -54.2128372, 31.6706219, -85.8933868, 85.9761810
7: -30.3680763, 39.5155869, -30.2397995, 39.3277855, -69.6958618, 69.7553864
8: -36.5265007, 53.4958878, -36.4481812, 53.4031677, -89.9296722, 89.9440689
9: -29.0477219, 38.9795609, -29.0242252, 38.9376869, -67.9854126, 68.0037842
10: -49.5023575, 43.7264328, -49.3943596, 43.6432533, -93.1456146, 93.1207886
11: -48.9208870, 21.8287544, -48.8828850, 21.8139210, -70.7348099, 70.7116394
12: -55.1626167, 25.0517426, -55.0309448, 24.9660301, -78.9986115, 78.9576416
13: -50.5076942, 43.7060966, -50.4901352, 43.7454033, -94.2530975, 94.1962280
14: -87.3638153, 31.0321140, -87.1201019, 30.8602829, -118.2240982, 118.1522141
15: -35.6915970, 35.9960289, -35.6398315, 36.0225220, -71.7141190, 71.6358643
16: -45.8293991, 33.7706985, -45.7654877, 33.6148262, -79.4442291, 79.5361862
17: -84.6928177, 23.3482189, -84.4426117, 23.2399368, -107.9327545, 107.7908325
18: -48.9018631, 31.2032280, -49.0671349, 31.2934456, -80.1953125, 80.2703629
19: -38.8318596, 18.4517365, -38.8589020, 18.5048866, -57.3367462, 57.3106384
20: -36.8338623, 23.2883606, -36.8221741, 23.2649117, -60.0987740, 60.1105347
21: -48.0015717, 21.9865494, -47.9720078, 21.9908752, -69.9924469, 69.9585571
22: -49.8757477, 22.0399208, -49.5860443, 21.9247379, -71.8004837, 71.6259613
23: -38.9903488, 23.9319592, -38.9113159, 23.8745823, -62.8649292, 62.8432770
24: -46.2359695, 23.9558029, -46.1652603, 23.9195976, -70.1555634, 70.1210632
25: -41.1455879, 24.7121830, -40.9840698, 24.5734444, -65.7190323, 65.6962509
26: -56.8338051, 33.3817635, -56.7414398, 33.3735733, -90.2073822, 90.1231995
27: -45.2059288, 28.6708755, -45.2681198, 28.6780968, -73.8840256, 73.9389954
28: -38.8382187, 26.6335354, -38.8013268, 26.5913582, -65.4295807, 65.4348602
29: -51.4982491, 20.5197029, -51.1669197, 20.3720169, -71.8702698, 71.6866226
30: -49.1636658, 26.0695210, -49.0749054, 25.9870224, -75.1506882, 75.1444244
31: -50.9399452, 27.7788105, -50.9932594, 27.7987061, -78.7386475, 78.7720718
32: -52.3066978, 24.5949574, -52.2677536, 24.5956421, -76.9023438, 76.8627090
33: -72.2117157, 33.7012787, -72.1631927, 33.6726799, -105.5538254, 105.4901047
34: -65.4456787, 17.0691071, -65.4098206, 17.0338650, -81.6202164, 81.5884018
35: -63.7186203, 23.5028515, -63.6344337, 23.4922905, -85.6512985, 85.5671997
36: -61.9269524, 24.3414593, -61.8462334, 24.2997227, -86.2266769, 86.1876907
37: -87.0684814, 19.7156944, -87.0146408, 19.5978527, -106.6663361, 106.7303314
38: -69.8674469, 29.1238365, -69.8008270, 28.9459476, -98.8133926, 98.9246674
39: -80.3332901, 30.5275288, -80.2211227, 30.3609810, -110.6942749, 110.7486496
40: -62.4918060, 25.5597458, -62.4215775, 25.2663174, -87.7581253, 87.9813232
41: -54.9354248, 32.7647705, -54.8374405, 32.6452370, -87.5806580, 87.6022110
42: -36.1706467, 25.9127750, -36.1811676, 25.9124603, -62.0831070, 62.0939407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=225, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 591
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1697115
time: 88.01 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9888773, upper bound: 44.2243924
time: 88.50 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -56.5039902, 43.4277802, -56.4810600, 43.3341408, -99.8381348, 99.9088440
1: -25.2521648, 37.7011490, -25.2249031, 37.6376190, -62.8897858, 62.9260521
2: -21.7509308, 37.1238861, -21.7465286, 37.1359787, -58.8869095, 58.8704147
3: -24.3624115, 39.7608032, -24.4295521, 39.8588867, -64.2212982, 64.1903534
4: -28.4197445, 43.6537552, -28.3667641, 43.5556946, -71.9754410, 72.0205231
5: -24.5787354, 39.6959534, -24.5849934, 39.6895905, -64.2683258, 64.2809448
6: -54.2239227, 31.7696228, -54.2650909, 31.7196426, -85.9435654, 86.0347137
7: -30.3694420, 39.5232391, -30.3057785, 39.3805237, -69.7499695, 69.8290176
8: -36.5272293, 53.5018387, -36.5063629, 53.4589615, -89.9861908, 90.0082016
9: -29.0450191, 38.9827499, -29.0523815, 38.9691467, -68.0141678, 68.0351334
10: -49.5066795, 43.7279816, -49.4422302, 43.7070312, -93.2137146, 93.1702118
11: -48.9289017, 21.8296242, -48.9541588, 21.8879623, -70.8168640, 70.7837830
12: -55.1665955, 25.0512657, -55.0751801, 24.9968510, -79.0363464, 78.9961624
13: -50.4996719, 43.7078934, -50.5019913, 43.7797012, -94.2793732, 94.2098846
14: -87.3742752, 31.0328922, -87.2066269, 30.9758911, -118.3501663, 118.2395172
15: -35.6911697, 35.9947205, -35.6647758, 36.0455093, -71.7366791, 71.6595001
16: -45.8322525, 33.7749176, -45.8476410, 33.6641693, -79.4964218, 79.6225586
17: -84.7069855, 23.3494186, -84.5470123, 23.3166981, -108.0236816, 107.8964310
18: -48.9037399, 31.1880150, -49.0878792, 31.2849960, -80.1887360, 80.2758942
19: -38.8370285, 18.4522858, -38.9078445, 18.5353546, -57.3723831, 57.3601303
20: -36.8366394, 23.2890873, -36.8538132, 23.3125229, -60.1491623, 60.1428986
21: -48.0090942, 21.9872398, -48.0361214, 22.0541973, -70.0632935, 70.0233612
22: -49.8890991, 22.0409584, -49.6763916, 22.0167503, -71.9058533, 71.7173462
23: -39.0012436, 23.9320583, -38.9857788, 23.9659348, -62.9671783, 62.9178391
24: -46.2435341, 23.9566536, -46.2241478, 23.9891930, -70.2327271, 70.1808014
25: -41.1566010, 24.7139130, -41.0554314, 24.6854782, -65.8420792, 65.7693481
26: -56.8372650, 33.3817024, -56.7861748, 33.4127960, -90.2500610, 90.1678772
27: -45.2078362, 28.6616707, -45.2993393, 28.6778831, -73.8857193, 73.9610138
28: -38.8454895, 26.6338043, -38.8581390, 26.6792431, -65.5247345, 65.4919434
29: -51.5161057, 20.5207596, -51.2870979, 20.5015984, -72.0177002, 71.8078613
30: -49.1774406, 26.0698624, -49.1498032, 26.1193047, -75.2967453, 75.2196655
31: -50.9448128, 27.7797451, -51.0383911, 27.8376236, -78.7824402, 78.8181381
32: -52.3013573, 24.5952644, -52.2866707, 24.6115303, -76.9128876, 76.8819351
33: -72.2132874, 33.7042007, -72.2142944, 33.7055435, -105.5769043, 105.6084061
34: -65.4480286, 17.0706520, -65.4489746, 17.0818577, -81.6439362, 81.6464996
35: -63.7187233, 23.5041046, -63.6674271, 23.5055313, -85.6563797, 85.6075668
36: -61.9276276, 24.3435459, -61.8937416, 24.3293514, -86.2569809, 86.2372894
37: -87.0709686, 19.7122459, -87.1308823, 19.6293621, -106.7003326, 106.8431244
38: -69.8677063, 29.1306000, -69.8935013, 29.0133972, -98.8811035, 99.0241013
39: -80.3341980, 30.5374203, -80.3096161, 30.4149017, -110.7490997, 110.8470383
40: -62.4934387, 25.5807610, -62.5667877, 25.3774071, -87.8708496, 88.1475525
41: -54.9372787, 32.7721519, -54.9244461, 32.6957703, -87.6330490, 87.6965942
42: -36.1640244, 25.9134598, -36.1884537, 25.9230728, -62.0870972, 62.1019135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=225, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 666
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1414
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1414
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 549
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9890241, upper bound: 44.1697115
time: 103.22 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2243924
time: 88.38 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -56.6961632, 43.5073471, -56.5175056, 43.3086700, -100.0048370, 100.0248566
1: -25.3959427, 37.7948227, -25.2625980, 37.6024666, -62.9984093, 63.0574188
2: -21.9643173, 37.2448311, -21.8047810, 37.1010437, -59.0653610, 59.0496140
3: -24.6067219, 39.9030762, -24.4970741, 39.7969589, -64.4036789, 64.4001465
4: -28.6543617, 43.7758255, -28.4330482, 43.4967880, -72.1511536, 72.2088776
5: -24.8044014, 39.8200378, -24.6460819, 39.6367722, -64.4411774, 64.4661179
6: -54.3091278, 31.8650513, -54.2541924, 31.7197151, -86.0288391, 86.1192474
7: -30.5575275, 39.5897598, -30.3544960, 39.3431168, -69.9006424, 69.9442596
8: -36.7163696, 53.6365280, -36.5572319, 53.4226799, -90.1390533, 90.1937561
9: -29.1463127, 39.0578499, -29.0690384, 38.9481354, -68.0944519, 68.1268921
10: -49.6419678, 43.9489365, -49.4560852, 43.7540131, -93.3959808, 93.4050217
11: -49.1644859, 22.1179371, -48.9147263, 21.9686832, -71.1331711, 71.0326614
12: -55.3169785, 25.3618774, -55.0361862, 25.0822239, -79.2559357, 79.2552795
13: -50.6503029, 43.8478889, -50.5320740, 43.7593079, -94.4096069, 94.3799591
14: -87.5799942, 31.3648949, -87.1778183, 31.0715771, -118.6515732, 118.5427094
15: -35.8510208, 36.0744476, -35.6863556, 36.0082474, -71.8592682, 71.7608032
16: -45.9768143, 33.9098015, -45.8703461, 33.6622772, -79.6390915, 79.7801514
17: -84.9171295, 23.6094856, -84.5138321, 23.3866119, -108.3037415, 108.1233215
18: -49.0942650, 31.4058495, -49.0613213, 31.3480873, -80.4423523, 80.4671707
19: -39.0073624, 18.6400375, -38.8762436, 18.5887928, -57.5961533, 57.5162811
20: -36.9788742, 23.4615364, -36.8377151, 23.3609467, -60.3398209, 60.2992516
21: -48.2170639, 22.2214336, -48.0136871, 22.1213112, -70.3383789, 70.2351227
22: -50.0023918, 22.1874657, -49.6539459, 22.0526962, -72.0550842, 71.8414154
23: -39.1439056, 24.1250381, -38.9390945, 24.0214119, -63.1653175, 63.0641327
24: -46.3642159, 24.0941734, -46.1785431, 24.0265179, -70.3907318, 70.2727203
25: -41.2603989, 24.8741169, -41.0103073, 24.7293854, -65.9897842, 65.8844223
26: -57.0341415, 33.7223587, -56.7690010, 33.5076828, -90.5418243, 90.4913635
27: -45.3650360, 28.7757645, -45.2741318, 28.7159100, -74.0809479, 74.0498962
28: -38.9916153, 26.8303680, -38.8222351, 26.7361794, -65.7277985, 65.6526031
29: -51.6529198, 20.7081985, -51.2512093, 20.5525742, -72.2054901, 71.9594116
30: -49.3367157, 26.3099327, -49.0968895, 26.1840992, -75.5208130, 75.4068222
31: -51.1424103, 27.9847717, -50.9919090, 27.8949871, -79.0373993, 78.9766846
32: -52.3989716, 24.6984329, -52.2786522, 24.6308270, -77.0298004, 76.9770813
33: -72.4084625, 33.8471336, -72.2560577, 33.7205887, -105.7904510, 105.7743835
34: -65.5317154, 17.1730499, -65.4429169, 17.1021194, -81.7600403, 81.7590332
35: -63.7909775, 23.5751667, -63.6756363, 23.5171947, -85.7678452, 85.7100983
36: -61.9961472, 24.4115334, -61.9056244, 24.3440990, -86.3402481, 86.3171539
37: -87.1746063, 19.8288765, -87.0931244, 19.6481972, -106.8227997, 106.9219971
38: -69.9856110, 29.2021027, -69.9164429, 29.0268726, -99.0124817, 99.1185455
39: -80.4886475, 30.6375904, -80.3319702, 30.4262047, -110.9148560, 110.9695587
40: -62.5944862, 25.6588173, -62.5482864, 25.3764610, -87.9709473, 88.2071075
41: -55.0218811, 32.8693275, -54.9263992, 32.6999969, -87.7218781, 87.7957306
42: -36.2331581, 26.0354519, -36.1728897, 25.9373074, -62.1704636, 62.2083435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=225, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=488, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 662
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 539
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 539
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 623
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 629
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 544
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 638

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9191626, upper bound: 44.1297748
time: 80.44 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9191626, upper bound: 44.1298869
time: 89.30 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -56.5019455, 43.4236565, -56.5239906, 43.3838158, -99.8857574, 99.9476471
1: -25.2503109, 37.6943817, -25.2596970, 37.6916313, -62.9419403, 62.9540787
2: -21.7495365, 37.1213379, -21.8023796, 37.1802521, -58.9297867, 58.9237175
3: -24.3667355, 39.7597122, -24.4728241, 39.9002266, -64.2669601, 64.2325363
4: -28.4182243, 43.6482620, -28.4451923, 43.6451263, -72.0633545, 72.0934525
5: -24.5786228, 39.6894150, -24.6554241, 39.7471085, -64.3257294, 64.3448410
6: -54.2227631, 31.7633457, -54.2763596, 31.7540226, -85.9767838, 86.0397034
7: -30.3680763, 39.5155869, -30.3851547, 39.4622726, -69.8303528, 69.9007416
8: -36.5265007, 53.4958878, -36.5587730, 53.5133133, -90.0398102, 90.0546570
9: -29.0477219, 38.9795609, -29.0791550, 38.9945145, -68.0422363, 68.0587158
10: -49.5023575, 43.7264328, -49.5276604, 43.8134079, -93.3157654, 93.2540894
11: -48.9208870, 21.8287544, -49.0264969, 21.9433289, -70.8642120, 70.8552551
12: -55.1626167, 25.0517426, -55.1974487, 25.1399479, -79.1620026, 79.1105347
13: -50.5076942, 43.7060966, -50.5581131, 43.8136749, -94.3213654, 94.2642059
14: -87.3638153, 31.0321140, -87.3978424, 31.1284828, -118.4922943, 118.4299545
15: -35.6915970, 35.9960289, -35.7365036, 36.0698967, -71.7614899, 71.7325287
16: -45.8293991, 33.7706985, -45.8720856, 33.7377243, -79.5671234, 79.6427841
17: -84.6928177, 23.3482189, -84.6953583, 23.4192924, -108.1121063, 108.0435791
18: -48.9018631, 31.2032280, -49.1032104, 31.2878265, -80.1896896, 80.3064423
19: -38.8318596, 18.4517365, -38.9308243, 18.5356026, -57.3674622, 57.3825607
20: -36.8338623, 23.2883606, -36.9017143, 23.3475857, -60.1814499, 60.1900749
21: -48.0015717, 21.9865494, -48.0797539, 22.0690403, -70.0706100, 70.0662994
22: -49.8757477, 22.0399208, -49.7772064, 22.0474434, -71.9231873, 71.8171234
23: -38.9903488, 23.9319592, -39.0325851, 23.9831982, -62.9735489, 62.9645462
24: -46.2359695, 23.9558029, -46.2793617, 24.0002003, -70.2361679, 70.2351685
25: -41.1455879, 24.7121830, -41.1291199, 24.7203121, -65.8658981, 65.8413010
26: -56.8338051, 33.3817635, -56.8973656, 33.5091858, -90.3429871, 90.2791290
27: -45.2059288, 28.6708755, -45.3354721, 28.7160530, -73.9219818, 74.0063477
28: -38.8382187, 26.6335354, -38.8969498, 26.6901054, -65.5283203, 65.5304871
29: -51.4982491, 20.5197029, -51.3866005, 20.5302334, -72.0284805, 71.9063034
30: -49.1636658, 26.0695210, -49.2021446, 26.1354179, -75.2990875, 75.2716675
31: -50.9399452, 27.7788105, -51.0810318, 27.8654022, -78.8053436, 78.8598404
32: -52.3066978, 24.5949574, -52.3363533, 24.6396141, -76.9463120, 76.9313126
33: -72.2117157, 33.7012787, -72.2840118, 33.7662964, -105.7560425, 105.7226639
34: -65.4456787, 17.0691071, -65.4650955, 17.0932827, -81.7816467, 81.7425919
35: -63.7186203, 23.5028515, -63.6963348, 23.5285797, -85.7819824, 85.7221069
36: -61.9269524, 24.3414593, -61.9159470, 24.3417873, -86.2687378, 86.2574081
37: -87.0684814, 19.7156944, -87.1512985, 19.6990204, -106.7675018, 106.8669891
38: -69.8674469, 29.1238365, -69.9043045, 29.0546780, -98.9221268, 99.0281372
39: -80.3332901, 30.5275288, -80.3721313, 30.4920845, -110.8253784, 110.8996582
40: -62.4918060, 25.5597458, -62.5385742, 25.4212112, -87.9130173, 88.0983200
41: -54.9354248, 32.7647705, -54.9557762, 32.7474670, -87.6828918, 87.7205505
42: -36.1706467, 25.9127750, -36.2030182, 25.9661064, -62.1367531, 62.1157913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=227, inp2_unstable=225, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=487, inp2_unstable=486, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 634
type: A, layer: 1, pos: 634
type: B, layer: 1, pos: 662
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 614
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 666
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 650
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 739
type: A, layer: 1, pos: 739
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 623
type: B, layer: 1, pos: 591
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1369
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: A, layer: 1, pos: 549
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 629
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 607
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 1369
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 969
type: B, layer: 1, pos: 969
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1696
type: A, layer: 1, pos: 1696
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: A, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 544
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 1464
type: A, layer: 1, pos: 1464

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 632

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -43.9890241, upper bound: 44.1699440
time: 90.20 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2246072
time: 102.31 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 195.01 seconds
IS_A2_B1_B1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9888773, upper bound: 44.1697115
IS_A2_B1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9888773, upper bound: 44.2243924
IS_A2_B1_B1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9890241, upper bound: 44.1697115
IS_A2_B1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2243924
IS_A2_B1_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9191626, upper bound: 44.1297748
IS_A2_B1_B1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9191626, upper bound: 44.1298869
IS_A2_B1_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9890241, upper bound: 44.1699440
IS_A2_B1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 195.01
Output dim: 4, lower bound: -43.9890241, upper bound: 44.2246072
IS_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 195.01
Output dim: 4, lower bound: -44.0313899, upper bound: 44.2308984
IS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 195.01
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2299835
IS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 195.01
Output dim: 4, lower bound: -43.9909015, upper bound: 44.2323576
IS_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 195.01
Output dim: 4, lower bound: -44.0496451, upper bound: 44.2361733
IS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 195.01
Output dim: 4, lower bound: -44.0022306, upper bound: 44.2363184

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 103.67 + 7268.75 = 7372.42 seconds
