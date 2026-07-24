## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 7200 seconds
Split limit: 100
Threshold: 38.181457323


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=249, inp2_unstable=249, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388)
1: (-26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461)
2: (-24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001)
3: (-28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680)
4: (-31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228)
5: (-28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674)
6: (-55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035)
7: (-32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505)
8: (-37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959)
9: (-30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573)
10: (-49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753)
11: (-48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688)
12: (-59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373871)
13: (-51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880)
14: (-79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897)
15: (-38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579)
16: (-48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783)
17: (-79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988)
18: (-48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538)
19: (-38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658)
20: (-34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885)
21: (-46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846)
22: (-49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286)
23: (-37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486)
24: (-45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513)
25: (-39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552)
26: (-56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825)
27: (-46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792)
28: (-37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667)
29: (-51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515)
30: (-46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646)
31: (-49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272)
32: (-55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1568069, 80.1567993)
33: (-73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0983200, 105.0983200)
34: (-63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2917633, 81.2917480)
35: (-60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6587524, 84.6587524)
36: (-60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885986)
37: (-89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701691, 107.9701843)
38: (-69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888)
39: (-83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197)
40: (-65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349915, 87.3349915)
41: (-58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127)
42: (-40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.01 + 160.80 = 163.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -38.2196770, upper bound: 38.2196770

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1721

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2177460, upper bound: 38.1720118
time: 194.05 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2177460, upper bound: 38.2177459
time: 90.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 284.90 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 284.90
Output dim: 2, lower bound: -38.2177460, upper bound: 38.1720118
IS_A2, status: Status.UNKNOWN, split count: 1, time: 284.90
Output dim: 2, lower bound: -38.2177460, upper bound: 38.2177459

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -57.1851006, 42.8572044, -57.2139587, 42.9137115, -100.0988159, 100.0711670
1: -26.5089455, 35.1934509, -26.5182838, 35.2472801, -61.7562256, 61.7117348
2: -24.3702145, 36.9335938, -24.3823204, 37.0017281, -61.3719406, 61.3159142
3: -28.6474590, 41.3369980, -28.6589031, 41.3916473, -70.0391083, 69.9959030
4: -31.6029854, 41.1803246, -31.6142769, 41.2473946, -72.8503799, 72.7946014
5: -28.0088768, 42.6057510, -28.0210800, 42.6700592, -70.6789398, 70.6268311
6: -55.1657677, 27.2177315, -55.2323952, 27.2350235, -82.4007874, 82.4501266
7: -32.4315567, 40.4364014, -32.4459419, 40.5020065, -72.9335632, 72.8823395
8: -37.1280899, 49.2813416, -37.1408005, 49.3659248, -86.4940186, 86.4221420
9: -30.2648926, 38.3359833, -30.2947521, 38.3517876, -68.6166840, 68.6307373
10: -49.5463867, 48.1370659, -49.5914268, 48.1569748, -97.7033615, 97.7284927
11: -48.5038147, 29.0568352, -48.5535660, 29.0751553, -77.5789719, 77.6103973
12: -59.7555771, 31.3566818, -59.8233833, 31.3825645, -90.7240295, 90.7643509
13: -51.3098831, 46.8914375, -51.3336334, 46.9779778, -98.2878571, 98.2250671
14: -79.5694122, 42.4505959, -79.5977402, 42.6047249, -122.1741333, 122.0483398
15: -38.1190567, 35.1428680, -38.1431122, 35.1703796, -73.2894363, 73.2859802
16: -48.5881500, 37.2168922, -48.6401596, 37.2333031, -85.8214569, 85.8570557
17: -79.5501251, 33.9710159, -79.5731506, 34.0813446, -113.6314697, 113.5441666
18: -48.2420044, 33.3889351, -48.2818108, 33.4048347, -81.6468353, 81.6707458
19: -38.2772522, 19.2546234, -38.3044205, 19.2643986, -57.5416489, 57.5590439
20: -34.7572937, 24.9411373, -34.7717209, 24.9724922, -59.7297859, 59.7128601
21: -46.2127724, 24.8810806, -46.2369156, 24.8946686, -71.1074371, 71.1179962
22: -49.1419487, 25.1595955, -49.1637115, 25.1808014, -74.3227539, 74.3233032
23: -37.8300552, 26.3309555, -37.8637047, 26.3480034, -64.1780548, 64.1946564
24: -45.4673347, 28.8294296, -45.4908104, 28.8496437, -74.3169785, 74.3202362
25: -39.7103882, 29.4034805, -39.7314453, 29.4434700, -69.1538544, 69.1349258
26: -55.9906616, 38.7295532, -56.0417137, 38.7504883, -94.7411499, 94.7712708
27: -46.0397224, 30.0947189, -46.0729828, 30.1071739, -76.1468964, 76.1677017
28: -36.9776764, 29.8305149, -37.0032692, 29.8510170, -66.8286896, 66.8337860
29: -51.2072372, 24.5661240, -51.2287865, 24.5778027, -75.7850418, 75.7949066
30: -46.3269501, 33.4220123, -46.3480644, 33.4437637, -79.7707138, 79.7700806
31: -49.1165390, 27.7521915, -49.1558380, 27.7703018, -76.8868408, 76.9080276
32: -55.5511551, 24.6161613, -55.6059875, 24.6337948, -80.0659332, 80.1017685
33: -73.7234955, 31.8261204, -73.7775726, 31.8399620, -105.0122528, 105.0507202
34: -63.6453133, 17.8802776, -63.7087784, 17.8916416, -81.1773224, 81.2382050
35: -60.8065262, 24.3797817, -60.8351898, 24.3862305, -84.6089096, 84.6303711
36: -60.8284378, 25.2877617, -60.8652039, 25.2998409, -86.1273956, 86.1520920
37: -89.3226318, 18.6034355, -89.4428635, 18.6166573, -107.7789078, 107.8890991
38: -69.7600555, 29.0530815, -69.7847900, 29.0749493, -98.8350067, 98.8378754
39: -83.3843079, 30.7592602, -83.4301300, 30.7755165, -114.1598206, 114.1893921
40: -65.6639099, 21.4766846, -65.7900162, 21.4922791, -87.1424332, 87.2489014
41: -58.6143303, 28.6501274, -58.7277298, 28.6658192, -87.2801514, 87.3778534
42: -40.1497269, 24.6800861, -40.2152176, 24.6967583, -64.8464813, 64.8953018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=249, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2143488, upper bound: 38.1043370
time: 104.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2145468, upper bound: 38.1687767
time: 82.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -57.3394165, 42.9502869, -57.2256279, 42.9381599, -100.2775726, 100.1759186
1: -26.6100540, 35.2832718, -26.5216255, 35.2707596, -61.8808136, 61.8048973
2: -24.4908791, 37.0423660, -24.3871098, 37.0315704, -61.5224495, 61.4294739
3: -28.7508850, 41.4262962, -28.6636238, 41.4149933, -70.1658783, 70.0899200
4: -31.7187576, 41.2955627, -31.6187096, 41.2763596, -72.9951172, 72.9142761
5: -28.1156330, 42.7112083, -28.0258923, 42.6980743, -70.8137054, 70.7370987
6: -55.2565041, 27.3304825, -55.2525558, 27.2419662, -82.4984741, 82.5830383
7: -32.5547638, 40.5423737, -32.4511642, 40.5294762, -73.0842438, 72.9935379
8: -37.2704239, 49.4209976, -37.1454811, 49.4029388, -86.6733627, 86.5664825
9: -30.3443737, 38.3968658, -30.3058891, 38.3578568, -68.7022324, 68.7027588
10: -49.6429749, 48.2242851, -49.6067696, 48.1648178, -97.8077927, 97.8310547
11: -48.6030998, 29.1229382, -48.5676231, 29.0827579, -77.6858597, 77.6905594
12: -59.8670120, 31.4958439, -59.8511124, 31.3932610, -90.8414383, 90.9304047
13: -51.4857864, 47.0326157, -51.3436584, 47.0128326, -98.4986191, 98.3762741
14: -79.8125458, 42.6819458, -79.6087418, 42.6726685, -122.4852142, 122.2906876
15: -38.2128868, 35.2187500, -38.1530685, 35.1804962, -73.3933868, 73.3718185
16: -48.6892624, 37.2767181, -48.6551170, 37.2398071, -85.9290695, 85.9318390
17: -79.7353821, 34.1525116, -79.5821381, 34.1292839, -113.8646698, 113.7346497
18: -48.3171005, 33.4961433, -48.2980919, 33.4111023, -81.7282028, 81.7942352
19: -38.3384857, 19.2934799, -38.3144951, 19.2683907, -57.6068764, 57.6079750
20: -34.8134155, 25.0006714, -34.7769661, 24.9847946, -59.7982101, 59.7776375
21: -46.2770119, 24.9274673, -46.2457809, 24.9000664, -71.1770782, 71.1732483
22: -49.2084579, 25.2299061, -49.1720695, 25.1873283, -74.3957825, 74.4019775
23: -37.9065475, 26.3860836, -37.8759804, 26.3550854, -64.2616348, 64.2620621
24: -45.5328140, 28.8719406, -45.5001869, 28.8546410, -74.3874512, 74.3721313
25: -39.7692337, 29.4666882, -39.7393303, 29.4526958, -69.2219315, 69.2060165
26: -56.0857086, 38.8492661, -56.0634155, 38.7589264, -94.8446350, 94.9126816
27: -46.1094627, 30.1760330, -46.0862350, 30.1122723, -76.2217331, 76.2622681
28: -37.0345039, 29.8851929, -37.0136642, 29.8590221, -66.8935242, 66.8988571
29: -51.2688255, 24.6245327, -51.2369156, 24.5823898, -75.8512115, 75.8614502
30: -46.3841515, 33.5004768, -46.3560486, 33.4529800, -79.8371277, 79.8565216
31: -49.2048950, 27.7997208, -49.1715508, 27.7755165, -76.9804077, 76.9712677
32: -55.6380043, 24.6973114, -55.6235123, 24.6408768, -80.1600342, 80.1974792
33: -73.8191376, 31.8950386, -73.7984848, 31.8450642, -105.1076355, 105.1430206
34: -63.7589569, 17.9941654, -63.7390060, 17.8956585, -81.2855682, 81.3980255
35: -60.8624763, 24.4162216, -60.8458519, 24.3880692, -84.6676788, 84.6734238
36: -60.9092636, 25.3620911, -60.8817406, 25.3045788, -86.2128830, 86.2431793
37: -89.5287933, 18.7763786, -89.4959030, 18.6218414, -107.9883957, 108.1203766
38: -69.8286591, 29.1022987, -69.7944412, 29.0827045, -98.9113617, 98.8967438
39: -83.4879532, 30.7947388, -83.4483185, 30.7779408, -114.2658920, 114.2430573
40: -65.8710785, 21.6857548, -65.8448563, 21.4982491, -87.3495865, 87.5116272
41: -58.7939758, 28.8417645, -58.7772827, 28.6717720, -87.4657440, 87.6190491
42: -40.2413025, 24.7869186, -40.2344971, 24.7033978, -64.9447021, 65.0214157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=249, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2143488, upper bound: 38.1505950
time: 80.55 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2145468, upper bound: 38.2145464
time: 125.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 208.10 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 208.10
Output dim: 2, lower bound: -38.2143488, upper bound: 38.1043370
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 208.10
Output dim: 2, lower bound: -38.2145468, upper bound: 38.1687767
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 208.10
Output dim: 2, lower bound: -38.2143488, upper bound: 38.1505950
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 208.10
Output dim: 2, lower bound: -38.2145468, upper bound: 38.2145464

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -57.1469154, 42.8392029, -57.0944862, 42.8569336, -100.0038452, 99.9336853
1: -26.4844837, 35.1828918, -26.4411602, 35.2139664, -61.6984482, 61.6240540
2: -24.3189087, 36.9234657, -24.2198429, 36.9697380, -61.2886467, 61.1433105
3: -28.6020260, 41.3165894, -28.5159245, 41.3274651, -69.9294891, 69.8325119
4: -31.5425625, 41.1646271, -31.4238167, 41.1978912, -72.7404556, 72.5884399
5: -27.9656792, 42.5856857, -27.8848991, 42.6067238, -70.5724030, 70.4705811
6: -55.1402855, 27.1758690, -55.1517715, 27.1088696, -82.2491531, 82.3276367
7: -32.3931885, 40.4218750, -32.3279610, 40.4560699, -72.8492584, 72.7498322
8: -37.0717735, 49.2637177, -36.9621925, 49.3103638, -86.3821411, 86.2259064
9: -30.2457733, 38.2750473, -30.2344513, 38.1587372, -68.4045105, 68.5094986
10: -49.5188370, 48.0146179, -49.5044022, 47.7684097, -97.2872467, 97.5190201
11: -48.4781532, 28.9772034, -48.4723015, 28.8226147, -77.3007660, 77.4495087
12: -59.7350159, 31.2357388, -59.7587051, 30.9988728, -90.3174438, 90.5770721
13: -51.2913971, 46.8330116, -51.2752228, 46.7940559, -98.0854492, 98.1082306
14: -79.5389328, 42.3422852, -79.5026016, 42.2614441, -121.8003769, 121.8448868
15: -38.0512085, 35.1204605, -37.9276276, 35.0995483, -73.1507568, 73.0480881
16: -48.5542030, 37.1448135, -48.5330772, 37.0048141, -85.5590210, 85.6778870
17: -79.5311356, 33.8860397, -79.5137787, 33.8144531, -113.3455887, 113.3998184
18: -48.2062721, 33.3641434, -48.1702347, 33.3264999, -81.5327759, 81.5343781
19: -38.2534103, 19.2452812, -38.2294388, 19.2349510, -57.4883614, 57.4747200
20: -34.7371902, 24.9155064, -34.7082367, 24.8915138, -59.6287041, 59.6237411
21: -46.1885910, 24.8539391, -46.1606750, 24.8091145, -70.9977036, 71.0146179
22: -49.0943451, 25.1349907, -49.0132446, 25.1024837, -74.1968307, 74.1482391
23: -37.8115158, 26.3177395, -37.8053970, 26.3062916, -64.1178055, 64.1231384
24: -45.4110680, 28.8194160, -45.3157845, 28.8182259, -74.2292938, 74.1352005
25: -39.6831818, 29.3860264, -39.6467133, 29.3881798, -69.0713654, 69.0327377
26: -55.9633675, 38.6785851, -55.9558372, 38.5907135, -94.5540771, 94.6344223
27: -45.9685898, 30.0845890, -45.8479576, 30.0751514, -76.0437393, 75.9325485
28: -36.9489136, 29.8165493, -36.9140549, 29.8069916, -66.7559052, 66.7306061
29: -51.1763000, 24.5373573, -51.1309357, 24.4877434, -75.6640472, 75.6682892
30: -46.3043404, 33.3834267, -46.2765884, 33.3276138, -79.6319580, 79.6600189
31: -49.0816841, 27.7378349, -49.0465508, 27.7249413, -76.8066254, 76.7843857
32: -55.5280190, 24.5664501, -55.5329514, 24.4793987, -79.8868790, 79.9774323
33: -73.6533356, 31.8069439, -73.5550995, 31.7793961, -104.8734207, 104.7833862
34: -63.6091118, 17.8631115, -63.5943527, 17.8374615, -81.0739594, 81.0697632
35: -60.7531624, 24.3650818, -60.6658592, 24.3398876, -84.5009460, 84.4239578
36: -60.7967110, 25.2746353, -60.7672386, 25.2584534, -86.0539398, 86.0406036
37: -89.2740402, 18.5857620, -89.2910309, 18.5607815, -107.6678619, 107.7094269
38: -69.7273483, 29.0346584, -69.6823273, 29.0169621, -98.7443085, 98.7169876
39: -83.3390503, 30.7477131, -83.2885056, 30.7389336, -114.0779877, 114.0362167
40: -65.6206512, 21.4657269, -65.6536026, 21.4587097, -87.0611725, 87.0929108
41: -58.5883751, 28.6343651, -58.6454887, 28.6169815, -87.2053528, 87.2798538
42: -40.1288338, 24.6231632, -40.1490173, 24.5261631, -64.6549988, 64.7721786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1818687, upper bound: 38.0997235
time: 111.22 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1818687, upper bound: 38.0997235
time: 78.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -57.1759491, 42.8521004, -57.2651138, 42.9521523, -100.1280975, 100.1172180
1: -26.5025368, 35.1918182, -26.5487232, 35.3034744, -61.8060112, 61.7405396
2: -24.3622856, 36.9313164, -24.4027729, 37.1163979, -61.4786835, 61.3340912
3: -28.6354485, 41.3323669, -28.6654682, 41.4863892, -70.1218414, 69.9978333
4: -31.5851517, 41.1767654, -31.6195297, 41.3991623, -72.9843140, 72.7962952
5: -27.9973812, 42.6016350, -28.0357914, 42.7639465, -70.7613297, 70.6374283
6: -55.1603546, 27.1998978, -55.3322372, 27.2332535, -82.3936081, 82.5321350
7: -32.4232903, 40.4334488, -32.5025826, 40.5485764, -72.9718628, 72.9360352
8: -37.1196594, 49.2779884, -37.1665497, 49.5073929, -86.6270523, 86.4445343
9: -30.2605400, 38.3274384, -30.4293098, 38.3720360, -68.6325760, 68.7567444
10: -49.5415573, 48.1213570, -49.9607468, 48.1621628, -97.7037201, 98.0821075
11: -48.4990654, 29.0463200, -48.8651657, 29.0748940, -77.5739594, 77.9114838
12: -59.7514000, 31.3414993, -60.1616058, 31.3993511, -90.7330322, 91.0901566
13: -51.3039284, 46.8820305, -51.3924255, 47.0305023, -98.3344269, 98.2744598
14: -79.5619965, 42.4378929, -79.8698654, 42.6065636, -122.1685638, 122.3077545
15: -38.1100349, 35.1389961, -38.1758423, 35.3396072, -73.4496460, 73.3148346
16: -48.5809021, 37.2067833, -48.8596344, 37.2415123, -85.8224182, 86.0664215
17: -79.5455475, 33.9613953, -79.8310242, 34.1224251, -113.6679688, 113.7924194
18: -48.2299805, 33.3810921, -48.3607941, 33.4287567, -81.6587372, 81.7418823
19: -38.2709045, 19.2520103, -38.3696404, 19.2833672, -57.5542717, 57.6216507
20: -34.7540283, 24.9366093, -34.8591232, 24.9773197, -59.7313461, 59.7957306
21: -46.2079659, 24.8750038, -46.3861732, 24.9059677, -71.1139374, 71.2611771
22: -49.1222839, 25.1545353, -49.1724548, 25.2595768, -74.3818588, 74.3269882
23: -37.8265305, 26.3231430, -37.9295044, 26.3541031, -64.1806335, 64.2526474
24: -45.4580841, 28.8263512, -45.5171013, 28.9151649, -74.3732452, 74.3434525
25: -39.7027855, 29.3994293, -39.7584763, 29.4849148, -69.1876984, 69.1579056
26: -55.9852409, 38.7172165, -56.1326904, 38.7687569, -94.7539978, 94.8499069
27: -46.0306473, 30.0911522, -46.1104317, 30.2164669, -76.2471161, 76.2015839
28: -36.9714966, 29.8255711, -37.0294571, 29.9114323, -66.8829269, 66.8550262
29: -51.1977997, 24.5599251, -51.2721252, 24.6026993, -75.8004990, 75.8320465
30: -46.3224869, 33.4112778, -46.4061203, 33.4587250, -79.7812119, 79.8173981
31: -49.1071510, 27.7475014, -49.2351341, 27.7833881, -76.8905411, 76.9826355
32: -55.5468521, 24.6083355, -55.7245636, 24.6434746, -80.0689468, 80.2148895
33: -73.7133484, 31.8227940, -73.7910156, 32.0413742, -105.2303696, 105.0443039
34: -63.6391907, 17.8758049, -63.7284470, 18.0271206, -81.3453217, 81.2311859
35: -60.7988510, 24.3773575, -60.8472748, 24.5824394, -84.8251953, 84.6246262
36: -60.8212051, 25.2849922, -60.8909531, 25.3816719, -86.2021561, 86.1748962
37: -89.3102112, 18.6000423, -89.4878769, 18.6794949, -107.8291702, 107.9239960
38: -69.7520294, 29.0486107, -69.8419342, 29.1429119, -98.8949432, 98.8905487
39: -83.3698425, 30.7567806, -83.4681778, 30.8918304, -114.2616730, 114.2249603
40: -65.6569138, 21.4701862, -65.8565674, 21.5268898, -87.1728058, 87.3056412
41: -58.6093292, 28.6431561, -58.7807732, 28.6798973, -87.2892303, 87.4239273
42: -40.1455078, 24.6598873, -40.3894348, 24.6896973, -64.8352051, 65.0493240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2087716, upper bound: 38.1357609
time: 79.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2087716, upper bound: 38.1637891
time: 75.58 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -57.3012199, 42.9322281, -57.1061440, 42.8813629, -100.1825867, 100.0383759
1: -26.5856094, 35.2727051, -26.4445190, 35.2374191, -61.8230286, 61.7172241
2: -24.4395714, 37.0322418, -24.2246094, 36.9995880, -61.4391594, 61.2568512
3: -28.7054539, 41.4058495, -28.5206070, 41.3507843, -70.0562363, 69.9264526
4: -31.6583424, 41.2797928, -31.4282684, 41.2268066, -72.8851471, 72.7080612
5: -28.0724220, 42.6911163, -27.8897266, 42.6347733, -70.7071991, 70.5808411
6: -55.2310028, 27.2887192, -55.1719513, 27.1158104, -82.3468170, 82.4606705
7: -32.5163803, 40.5278091, -32.3331223, 40.4835434, -72.9999237, 72.8609314
8: -37.2140656, 49.4033203, -36.9668655, 49.3473740, -86.5614395, 86.3701859
9: -30.3252163, 38.3359451, -30.2455349, 38.1647911, -68.4900055, 68.5814819
10: -49.6153336, 48.1018600, -49.5195541, 47.7762794, -97.3916168, 97.6214142
11: -48.5774193, 29.0433044, -48.4864311, 28.8302269, -77.4076462, 77.5297394
12: -59.8463516, 31.3750000, -59.7864532, 31.0095978, -90.4347839, 90.7432251
13: -51.4672890, 46.9741669, -51.2851868, 46.8289185, -98.2962036, 98.2593536
14: -79.7820740, 42.5736084, -79.5135803, 42.3292999, -122.1113739, 122.0871887
15: -38.1450424, 35.1963348, -37.9375610, 35.1097183, -73.2547607, 73.1338959
16: -48.6553001, 37.2046013, -48.5480919, 37.0113029, -85.6666031, 85.7526932
17: -79.7163544, 34.0675964, -79.5227814, 33.8623581, -113.5787125, 113.5903778
18: -48.2813263, 33.4713364, -48.1864662, 33.3327637, -81.6140900, 81.6578064
19: -38.3146248, 19.2841244, -38.2395287, 19.2389297, -57.5535545, 57.5236511
20: -34.7933273, 24.9750404, -34.7134552, 24.9038239, -59.6971512, 59.6884956
21: -46.2529373, 24.9003677, -46.1696243, 24.8144894, -71.0674286, 71.0699921
22: -49.1608429, 25.2052784, -49.0216293, 25.1089897, -74.2698364, 74.2269058
23: -37.8879700, 26.3728771, -37.8177261, 26.3133678, -64.2013397, 64.1906052
24: -45.4766769, 28.8619556, -45.3251457, 28.8232346, -74.2999115, 74.1871033
25: -39.7420654, 29.4492168, -39.6545639, 29.3974094, -69.1394730, 69.1037827
26: -56.0584297, 38.7982178, -55.9775352, 38.5990753, -94.6575012, 94.7757568
27: -46.0383148, 30.1659279, -45.8612175, 30.0802441, -76.1185608, 76.0271454
28: -37.0056915, 29.8712196, -36.9244385, 29.8149872, -66.8206787, 66.7956543
29: -51.2378616, 24.5957623, -51.1390533, 24.4923210, -75.7301788, 75.7348175
30: -46.3614807, 33.4618950, -46.2845154, 33.3368378, -79.6983185, 79.7464142
31: -49.1700134, 27.7853737, -49.0622749, 27.7301216, -76.9001312, 76.8476486
32: -55.6148720, 24.6476460, -55.5504761, 24.4864101, -79.9808655, 80.0732269
33: -73.7489548, 31.8758373, -73.5759506, 31.7844887, -104.9687195, 104.8756104
34: -63.7227516, 17.9770184, -63.6245880, 17.8414917, -81.1821518, 81.2295990
35: -60.8090706, 24.4015007, -60.6764832, 24.3417320, -84.5597305, 84.4669876
36: -60.8775330, 25.3490257, -60.7837830, 25.2631798, -86.1394272, 86.1317215
37: -89.4801636, 18.7586727, -89.3440781, 18.5659809, -107.8772430, 107.9405518
38: -69.7959290, 29.0838852, -69.6919785, 29.0247784, -98.8207092, 98.7758636
39: -83.4426346, 30.7831173, -83.3065338, 30.7413692, -114.1840057, 114.0896530
40: -65.8278046, 21.6748238, -65.7083740, 21.4646683, -87.2681503, 87.3556213
41: -58.7679825, 28.8260384, -58.6950760, 28.6229458, -87.3909302, 87.5211182
42: -40.2203827, 24.7299919, -40.1682816, 24.5327911, -64.7531738, 64.8982697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1818687, upper bound: 38.1455225
time: 111.36 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1818687, upper bound: 38.1455225
time: 80.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -57.3302078, 42.9451790, -57.2768250, 42.9765892, -100.3067932, 100.2220001
1: -26.6036453, 35.2816162, -26.5521202, 35.3269348, -61.9305801, 61.8337364
2: -24.4829540, 37.0401039, -24.4075737, 37.1461716, -61.6291275, 61.4476776
3: -28.7388878, 41.4216576, -28.6701622, 41.5096893, -70.2485809, 70.0918198
4: -31.7009220, 41.2919769, -31.6239834, 41.4279861, -73.1289062, 72.9159622
5: -28.1041298, 42.7070541, -28.0406113, 42.7919235, -70.8960571, 70.7476654
6: -55.2510872, 27.3126717, -55.3523788, 27.2402401, -82.4913254, 82.6650543
7: -32.5465813, 40.5394058, -32.5078239, 40.5760002, -73.1225815, 73.0472260
8: -37.2619553, 49.4176483, -37.1712418, 49.5442886, -86.8062439, 86.5888901
9: -30.3400383, 38.3883095, -30.4403687, 38.3781128, -68.7181549, 68.8286743
10: -49.6381531, 48.2085533, -49.9758682, 48.1700401, -97.8081970, 98.1844177
11: -48.5983772, 29.1124001, -48.8792000, 29.0824966, -77.6808777, 77.9916000
12: -59.8628273, 31.4807167, -60.1893044, 31.4101677, -90.8505402, 91.2561493
13: -51.4797935, 47.0231628, -51.4024277, 47.0653381, -98.5451355, 98.4255905
14: -79.8051605, 42.6692390, -79.8808441, 42.6744232, -122.4795837, 122.5500793
15: -38.2038803, 35.2148781, -38.1858025, 35.3496933, -73.5535736, 73.4006805
16: -48.6820183, 37.2666092, -48.8745193, 37.2480469, -85.9300690, 86.1411285
17: -79.7307816, 34.1429138, -79.8400421, 34.1703491, -113.9011307, 113.9829559
18: -48.3050461, 33.4883118, -48.3770218, 33.4350662, -81.7401123, 81.8653336
19: -38.3321037, 19.2908669, -38.3797150, 19.2873287, -57.6194305, 57.6705818
20: -34.8101883, 24.9961395, -34.8643494, 24.9896507, -59.7998390, 59.8604889
21: -46.2722397, 24.9213676, -46.3949699, 24.9113636, -71.1836014, 71.3163376
22: -49.1887932, 25.2248306, -49.1808243, 25.2661018, -74.4548950, 74.4056549
23: -37.9030228, 26.3782806, -37.9417763, 26.3611755, -64.2641983, 64.3200531
24: -45.5236092, 28.8688774, -45.5264893, 28.9201527, -74.4437637, 74.3953705
25: -39.7616272, 29.4626389, -39.7663422, 29.4941483, -69.2557755, 69.2289810
26: -56.0802689, 38.8369408, -56.1543999, 38.7771912, -94.8574600, 94.9913406
27: -46.1003647, 30.1724892, -46.1236992, 30.2215652, -76.3219299, 76.2961884
28: -37.0283051, 29.8802395, -37.0398254, 29.9194393, -66.9477463, 66.9200668
29: -51.2593842, 24.6183624, -51.2802658, 24.6072922, -75.8666763, 75.8986282
30: -46.3796654, 33.4897118, -46.4140015, 33.4679451, -79.8476105, 79.9037170
31: -49.1954803, 27.7950478, -49.2507706, 27.7885742, -76.9840546, 77.0458221
32: -55.6336975, 24.6894798, -55.7420616, 24.6505966, -80.1631165, 80.3105469
33: -73.8090057, 31.8917503, -73.8119583, 32.0464935, -105.3257446, 105.1366348
34: -63.7528152, 17.9897079, -63.7586594, 18.0311394, -81.4535599, 81.3909988
35: -60.8547745, 24.4138107, -60.8579636, 24.5842819, -84.8840027, 84.6677246
36: -60.9020271, 25.3593674, -60.9075012, 25.3862858, -86.2875824, 86.2659912
37: -89.5163574, 18.7730103, -89.5409088, 18.6847095, -108.0386353, 108.1551590
38: -69.8205872, 29.0978413, -69.8516006, 29.1506233, -98.9712067, 98.9494400
39: -83.4734802, 30.7921734, -83.4862518, 30.8942833, -114.3677673, 114.2784271
40: -65.8640594, 21.6792393, -65.9113312, 21.5328751, -87.3799438, 87.5683060
41: -58.7889557, 28.8348141, -58.8302765, 28.6858826, -87.4748383, 87.6650925
42: -40.2370758, 24.7667122, -40.4086304, 24.6963310, -64.9334106, 65.1753387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1819887, upper bound: 38.2087713
time: 98.91 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2087716, upper bound: 38.2087713
time: 93.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 194.66 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.1818687, upper bound: 38.0997235
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.1818687, upper bound: 38.0997235
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.2087716, upper bound: 38.1357609
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.2087716, upper bound: 38.1637891
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.1818687, upper bound: 38.1455225
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.1818687, upper bound: 38.1455225
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.1819887, upper bound: 38.2087713
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 194.66
Output dim: 2, lower bound: -38.2087716, upper bound: 38.2087713

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -57.1296616, 42.8075180, -56.9193726, 42.7489853, -99.8786469, 99.7268906
1: -26.4787121, 35.1514549, -26.3172398, 35.1055717, -61.5842819, 61.4686966
2: -24.3119717, 36.8915863, -24.1050816, 36.8611641, -61.1731339, 60.9966660
3: -28.5950317, 41.2635422, -28.3508110, 41.1516800, -69.7467117, 69.6143494
4: -31.5353966, 41.1297531, -31.3020954, 41.0732269, -72.6086273, 72.4318466
5: -27.9586220, 42.5418816, -27.7422314, 42.4592361, -70.4178619, 70.2841110
6: -55.1113167, 27.1659737, -55.0385666, 27.0240002, -82.1353149, 82.2045441
7: -32.3841553, 40.3774490, -32.1559372, 40.3111076, -72.6952667, 72.5333862
8: -37.0634995, 49.2082825, -36.7841187, 49.1180992, -86.1815948, 85.9924011
9: -30.2336273, 38.2661896, -30.1563301, 38.1060867, -68.3397141, 68.4225159
10: -49.4835014, 48.0037460, -49.3475227, 47.6477165, -97.1312180, 97.3512726
11: -48.4545784, 28.9709339, -48.3274193, 28.7810822, -77.2356567, 77.2983551
12: -59.7028046, 31.2183895, -59.6478729, 30.8512764, -90.1338501, 90.4464951
13: -51.2789612, 46.7805824, -51.1300468, 46.6105118, -97.8894730, 97.9106293
14: -79.5231171, 42.2564316, -79.2439880, 41.9884682, -121.5115814, 121.5004196
15: -38.0395813, 35.1020393, -37.8431892, 35.0020752, -73.0416565, 72.9452286
16: -48.5213318, 37.1375694, -48.3701248, 36.9545021, -85.4758301, 85.5076904
17: -79.5190125, 33.8195496, -79.2681046, 33.5871620, -113.1061707, 113.0876541
18: -48.1766205, 33.3551521, -48.0481453, 33.2062531, -81.3828735, 81.4032974
19: -38.2250214, 19.2412071, -38.1123657, 19.1756840, -57.4007034, 57.3535728
20: -34.7282257, 24.8957977, -34.6381454, 24.8200188, -59.5482445, 59.5339432
21: -46.1717644, 24.8484135, -46.0663033, 24.7683506, -70.9401169, 70.9147186
22: -49.0838013, 25.1224537, -48.9537811, 25.0150757, -74.0988770, 74.0762329
23: -37.7885208, 26.3117580, -37.6989670, 26.2542267, -64.0427475, 64.0107269
24: -45.3868942, 28.8150520, -45.2164497, 28.7737083, -74.1605988, 74.0315018
25: -39.6694412, 29.3763008, -39.5765343, 29.3255138, -68.9949570, 68.9528351
26: -55.9497032, 38.6645622, -55.8889389, 38.4846382, -94.4343414, 94.5534973
27: -45.9515800, 30.0775166, -45.7650909, 30.0263233, -75.9779053, 75.8426056
28: -36.9382362, 29.8006420, -36.8508606, 29.7339058, -66.6721420, 66.6515045
29: -51.1662025, 24.5304718, -51.0562859, 24.4328423, -75.5990448, 75.5867615
30: -46.2918243, 33.3658218, -46.1929932, 33.2413521, -79.5331726, 79.5588150
31: -49.0409508, 27.7323799, -48.8856850, 27.6482105, -76.6891632, 76.6180649
32: -55.5033417, 24.5550690, -55.4312325, 24.3977623, -79.7812653, 79.8630219
33: -73.6086960, 31.7987633, -73.3998108, 31.6508160, -104.6880417, 104.6114426
34: -63.5943604, 17.8551140, -63.5285988, 17.7565746, -80.9768066, 80.9931488
35: -60.7404327, 24.3599072, -60.6072121, 24.2770042, -84.4194946, 84.3544617
36: -60.7884941, 25.2667294, -60.7179871, 25.1944180, -85.9813690, 85.9832535
37: -89.1981888, 18.5787029, -89.0285721, 18.3447304, -107.3725357, 107.4348526
38: -69.7109756, 29.0185032, -69.5826111, 28.9216976, -98.6326752, 98.6011124
39: -83.2896957, 30.7409916, -83.0987015, 30.6158066, -113.9055023, 113.8396912
40: -65.5567627, 21.4577179, -65.4264679, 21.2503242, -86.7857513, 86.8550949
41: -58.5340347, 28.6266041, -58.4577141, 28.4589291, -86.9929657, 87.0843201
42: -40.0965767, 24.6135769, -40.0268707, 24.4194469, -64.5160217, 64.6404495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1482238, upper bound: 38.0990002
time: 79.70 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1482238, upper bound: 38.0990002
time: 93.29 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -57.1441612, 42.8363342, -57.0861816, 42.8483696, -99.9925308, 99.9225159
1: -26.4832478, 35.1799469, -26.4375153, 35.2050629, -61.6883087, 61.6174622
2: -24.3176041, 36.9204750, -24.2160702, 36.9606705, -61.2782745, 61.1365433
3: -28.6007347, 41.3116226, -28.5122128, 41.3123245, -69.9130554, 69.8238373
4: -31.5413456, 41.1607018, -31.4202557, 41.1860199, -72.7273636, 72.5809555
5: -27.9646244, 42.5816765, -27.8817978, 42.5946198, -70.5592422, 70.4634705
6: -55.1324997, 27.1743164, -55.1297531, 27.1042271, -82.2367249, 82.3040695
7: -32.3917999, 40.4179459, -32.3239441, 40.4441452, -72.8359451, 72.7418900
8: -37.0703201, 49.2592316, -36.9578552, 49.2966957, -86.3670197, 86.2170868
9: -30.2421932, 38.2735100, -30.2234726, 38.1543121, -68.3965073, 68.4969788
10: -49.5108528, 48.0124130, -49.4798965, 47.7619553, -97.2728119, 97.4923096
11: -48.4724236, 28.9757824, -48.4547997, 28.8184509, -77.2908783, 77.4305801
12: -59.7282181, 31.2334290, -59.7401505, 30.9920959, -90.3038330, 90.5505066
13: -51.2893143, 46.8290100, -51.2690582, 46.7833252, -98.0726395, 98.0980682
14: -79.5353546, 42.3358765, -79.4919052, 42.2420044, -121.7773590, 121.8277817
15: -38.0494118, 35.1153069, -37.9223251, 35.0836449, -73.1330566, 73.0376282
16: -48.5438156, 37.1432571, -48.5010796, 37.0002289, -85.5440445, 85.6443329
17: -79.5279388, 33.8793030, -79.5043945, 33.7958679, -113.3238068, 113.3836975
18: -48.2028923, 33.3620911, -48.1607971, 33.3204041, -81.5233002, 81.5228882
19: -38.2473907, 19.2439251, -38.2111168, 19.2308998, -57.4782906, 57.4550400
20: -34.7355194, 24.9121704, -34.7032814, 24.8816814, -59.6172028, 59.6154518
21: -46.1845284, 24.8526039, -46.1485481, 24.8050632, -70.9895935, 71.0011520
22: -49.0922966, 25.1317749, -49.0072556, 25.0925426, -74.1848373, 74.1390305
23: -37.8087578, 26.3168163, -37.7970428, 26.3036137, -64.1123734, 64.1138611
24: -45.4078026, 28.8183823, -45.3062553, 28.8151722, -74.2229767, 74.1246338
25: -39.6810112, 29.3834763, -39.6402588, 29.3804703, -69.0614777, 69.0237350
26: -55.9606400, 38.6743546, -55.9479370, 38.5777206, -94.5383606, 94.6222916
27: -45.9655571, 30.0832253, -45.8389931, 30.0710506, -76.0366058, 75.9222183
28: -36.9472580, 29.8137550, -36.9091034, 29.7984943, -66.7457504, 66.7228546
29: -51.1741028, 24.5363617, -51.1243935, 24.4848137, -75.6589203, 75.6607513
30: -46.3020210, 33.3780594, -46.2696342, 33.3139763, -79.6159973, 79.6476898
31: -49.0721016, 27.7361298, -49.0184250, 27.7198353, -76.7919388, 76.7545547
32: -55.5221672, 24.5644913, -55.5168266, 24.4735584, -79.8746338, 79.9562378
33: -73.6489944, 31.8057842, -73.5419769, 31.7758617, -104.8647690, 104.7535324
34: -63.6058807, 17.8611488, -63.5848618, 17.8315659, -81.0638275, 81.0467682
35: -60.7506104, 24.3640423, -60.6582756, 24.3367100, -84.4941254, 84.4053345
36: -60.7949486, 25.2724762, -60.7620773, 25.2519836, -86.0456238, 86.0331726
37: -89.2669373, 18.5849018, -89.2694473, 18.5582924, -107.6576080, 107.6793365
38: -69.7249146, 29.0280800, -69.6751251, 28.9968128, -98.7217255, 98.7032013
39: -83.3341827, 30.7466698, -83.2738876, 30.7358685, -114.0700531, 114.0205536
40: -65.6152954, 21.4646053, -65.6373596, 21.4553986, -87.0520172, 87.0710831
41: -58.5837135, 28.6327438, -58.6313934, 28.6121941, -87.1959076, 87.2641373
42: -40.1233368, 24.6217880, -40.1342239, 24.5221844, -64.6455231, 64.7560120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1753361, upper bound: 38.0990002
time: 75.09 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1482238, upper bound: 38.0990002
time: 109.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -57.0007286, 42.7440186, -57.2477074, 42.9205399, -99.9212646, 99.9917297
1: -26.3786316, 35.0834122, -26.5428410, 35.2721214, -61.6507530, 61.6262512
2: -24.2475262, 36.8226547, -24.3957996, 37.0846062, -61.3321304, 61.2184525
3: -28.4702682, 41.1565933, -28.6584435, 41.4334335, -69.9037018, 69.8150330
4: -31.4634590, 41.0520134, -31.6122971, 41.3644371, -72.8278961, 72.6643066
5: -27.8545856, 42.4539948, -28.0286694, 42.7202988, -70.5748825, 70.4826660
6: -55.0468788, 27.1152992, -55.3034973, 27.2232285, -82.2701111, 82.4187927
7: -32.2512054, 40.2882729, -32.4933968, 40.5042419, -72.7554474, 72.7816696
8: -36.9415627, 49.0857544, -37.1582108, 49.4521866, -86.3937531, 86.2439651
9: -30.1823196, 38.2747612, -30.4171600, 38.3631973, -68.5455170, 68.6919250
10: -49.3839798, 48.0006905, -49.9259758, 48.1512451, -97.5352249, 97.9266663
11: -48.3536949, 29.0048561, -48.8417130, 29.0685768, -77.4222717, 77.8465729
12: -59.6404648, 31.1939449, -60.1293945, 31.3818970, -90.6024704, 90.9065399
13: -51.1586990, 46.6980553, -51.3799171, 46.9780464, -98.1367493, 98.0779724
14: -79.3032532, 42.1650391, -79.8540344, 42.5206223, -121.8238754, 122.0190735
15: -38.0256996, 35.0414429, -38.1641464, 35.3212280, -73.3469238, 73.2055893
16: -48.4174919, 37.1563683, -48.8269577, 37.2342224, -85.6517181, 85.9833221
17: -79.2997818, 33.7332039, -79.8188477, 34.0558777, -113.3556595, 113.5520477
18: -48.1076965, 33.2606735, -48.3311424, 33.4197311, -81.5274277, 81.5918121
19: -38.1534920, 19.1928577, -38.3413086, 19.2792645, -57.4327545, 57.5341644
20: -34.6839409, 24.8651257, -34.8502045, 24.9575996, -59.6415405, 59.7153320
21: -46.1132965, 24.8342285, -46.3696594, 24.9003830, -71.0136795, 71.2038879
22: -49.0627899, 25.0672016, -49.1618500, 25.2469978, -74.3097839, 74.2290497
23: -37.7200241, 26.2711220, -37.9066353, 26.3481636, -64.0681915, 64.1777573
24: -45.3586311, 28.7818203, -45.4929466, 28.9107857, -74.2694168, 74.2747650
25: -39.6325951, 29.3368015, -39.7446976, 29.4751740, -69.1077728, 69.0814972
26: -55.9182816, 38.6113892, -56.1190834, 38.7547073, -94.6729889, 94.7304688
27: -45.9475288, 30.0423203, -46.0933533, 30.2094326, -76.1569595, 76.1356735
28: -36.9082642, 29.7524376, -37.0187225, 29.8955593, -66.8038254, 66.7711639
29: -51.1231651, 24.5050430, -51.2619362, 24.5957794, -75.7189484, 75.7669830
30: -46.2391052, 33.3250809, -46.3936653, 33.4410706, -79.6801758, 79.7187500
31: -48.9458733, 27.6707401, -49.1946030, 27.7778893, -76.7237625, 76.8653412
32: -55.4450455, 24.5268631, -55.6999207, 24.6319771, -79.9542770, 80.1093445
33: -73.5581055, 31.6942539, -73.7464142, 32.0331573, -105.0584106, 104.8589935
34: -63.5734634, 17.7948875, -63.7136765, 18.0191956, -81.2687988, 81.1339035
35: -60.7402573, 24.3145065, -60.8344765, 24.5772877, -84.7557983, 84.5431137
36: -60.7720032, 25.2208328, -60.8826294, 25.3738899, -86.1450348, 86.1021576
37: -89.0477676, 18.3840561, -89.4119568, 18.6724586, -107.5545349, 107.6286011
38: -69.6522598, 28.9532547, -69.8255157, 29.1267357, -98.7789917, 98.7787704
39: -83.1801605, 30.6335926, -83.4187393, 30.8851242, -114.0652847, 114.0523300
40: -65.4297333, 21.2617760, -65.7926331, 21.5188408, -86.9348907, 87.0300598
41: -58.4214439, 28.4850426, -58.7264557, 28.6721001, -87.0935440, 87.2115021
42: -40.0232239, 24.5532665, -40.3573608, 24.6800556, -64.7032776, 64.9106293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1350350
time: 91.58 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1350350
time: 78.75 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -57.1675568, 42.8435440, -57.2623138, 42.9492950, -100.1168518, 100.1058578
1: -26.4988861, 35.1829414, -26.5474873, 35.3005409, -61.7994270, 61.7304306
2: -24.3584538, 36.9222374, -24.4014835, 37.1134148, -61.4718704, 61.3237228
3: -28.6317139, 41.3172607, -28.6641808, 41.4814148, -70.1131287, 69.9814453
4: -31.5815716, 41.1649246, -31.6182899, 41.3952942, -72.9768677, 72.7832184
5: -27.9942474, 42.5895157, -28.0347080, 42.7599640, -70.7542114, 70.6242218
6: -55.1384430, 27.1952667, -55.3243866, 27.2316818, -82.3701248, 82.5196533
7: -32.4192123, 40.4215317, -32.5011597, 40.5446777, -72.9638901, 72.9226913
8: -37.1153221, 49.2643127, -37.1650543, 49.5029297, -86.6182556, 86.4293671
9: -30.2495098, 38.3229485, -30.4257240, 38.3705215, -68.6200333, 68.7486725
10: -49.5170288, 48.1148376, -49.9528160, 48.1599617, -97.6769867, 98.0676575
11: -48.4815292, 29.0421371, -48.8594513, 29.0734653, -77.5549927, 77.9015884
12: -59.7329254, 31.3347092, -60.1548080, 31.3970413, -90.7064819, 91.0764236
13: -51.2977142, 46.8712845, -51.3903542, 47.0265198, -98.3242340, 98.2616425
14: -79.5512543, 42.4184723, -79.8662872, 42.6001205, -122.1513748, 122.2847595
15: -38.1047287, 35.1233063, -38.1740570, 35.3343811, -73.4391098, 73.2973633
16: -48.5489388, 37.2021713, -48.8492012, 37.2399635, -85.7889023, 86.0513763
17: -79.5361099, 33.9429131, -79.8278351, 34.1156006, -113.6517105, 113.7707520
18: -48.2205696, 33.3749390, -48.3574066, 33.4266968, -81.6472626, 81.7323456
19: -38.2525558, 19.2479744, -38.3636246, 19.2820129, -57.5345688, 57.6115990
20: -34.7490692, 24.9267368, -34.8574715, 24.9739647, -59.7230339, 59.7842102
21: -46.1956863, 24.8709164, -46.3821678, 24.9046059, -71.1002960, 71.2530823
22: -49.1162720, 25.1447277, -49.1704216, 25.2563477, -74.3726196, 74.3151474
23: -37.8181839, 26.3204937, -37.9267502, 26.3531990, -64.1713867, 64.2472458
24: -45.4485054, 28.8232632, -45.5138397, 28.9141235, -74.3626251, 74.3371048
25: -39.6962929, 29.3917656, -39.7563057, 29.4823551, -69.1786499, 69.1480713
26: -55.9772415, 38.7044487, -56.1300087, 38.7644882, -94.7417297, 94.8344574
27: -46.0215187, 30.0870514, -46.1073837, 30.2150822, -76.2366028, 76.1944351
28: -36.9665451, 29.8170776, -37.0278091, 29.9086418, -66.8751831, 66.8448868
29: -51.1912460, 24.5569878, -51.2699165, 24.6016998, -75.7929459, 75.8269043
30: -46.3155136, 33.3976822, -46.4037971, 33.4532776, -79.7687912, 79.8014832
31: -49.0789146, 27.7424049, -49.2255554, 27.7816734, -76.8605881, 76.9679565
32: -55.5308113, 24.6024551, -55.7185974, 24.6415024, -80.0479279, 80.2025681
33: -73.7002258, 31.8192749, -73.7866974, 32.0402184, -105.2005310, 105.0356903
34: -63.6297226, 17.8699074, -63.7252045, 18.0251579, -81.3223648, 81.2210693
35: -60.7912445, 24.3741779, -60.8447342, 24.5813866, -84.8065414, 84.6178436
36: -60.8160133, 25.2784615, -60.8891907, 25.3795395, -86.1947327, 86.1665268
37: -89.2886124, 18.5975399, -89.4807205, 18.6786327, -107.7989655, 107.9137573
38: -69.7447891, 29.0284195, -69.8394775, 29.1363392, -98.8811264, 98.8678970
39: -83.3552246, 30.7537212, -83.4632797, 30.8908157, -114.2460403, 114.2170029
40: -65.6406860, 21.4668083, -65.8511963, 21.5257607, -87.1509781, 87.2964630
41: -58.5951805, 28.6383514, -58.7760925, 28.6782761, -87.2734528, 87.4144440
42: -40.1309090, 24.6558800, -40.3838425, 24.6883202, -64.8192291, 65.0397186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=248, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1630668
time: 82.85 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1630668
time: 81.06 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -57.2839890, 42.9005241, -56.9311409, 42.7733955, -100.0573883, 99.8316650
1: -26.5798721, 35.2411766, -26.3206558, 35.1289902, -61.7088623, 61.5618324
2: -24.4326477, 37.0003166, -24.1099052, 36.8909607, -61.3236084, 61.1102219
3: -28.6985016, 41.3526993, -28.3555260, 41.1749496, -69.8734512, 69.7082214
4: -31.6512337, 41.2448730, -31.3065681, 41.1021118, -72.7533417, 72.5514374
5: -28.0653801, 42.6472321, -27.7470837, 42.4871902, -70.5525665, 70.3943176
6: -55.2020416, 27.2789116, -55.0586929, 27.0309582, -82.2330017, 82.3376007
7: -32.5073853, 40.4833450, -32.1612511, 40.3385353, -72.8459167, 72.6445923
8: -37.2058258, 49.3477402, -36.7888336, 49.1549911, -86.3608170, 86.1365738
9: -30.3128986, 38.3271637, -30.1672554, 38.1121864, -68.4250870, 68.4944153
10: -49.5797844, 48.0910568, -49.3625183, 47.6555939, -97.2353821, 97.4535751
11: -48.5537834, 29.0370350, -48.3415642, 28.7886963, -77.3424835, 77.3786011
12: -59.8138351, 31.3578453, -59.6753731, 30.8621254, -90.2510223, 90.6125031
13: -51.4548035, 46.9216270, -51.1400642, 46.6453476, -98.1001511, 98.0616913
14: -79.7663040, 42.4876785, -79.2550201, 42.0562935, -121.8226013, 121.7426987
15: -38.1334305, 35.1778870, -37.8531418, 35.0122452, -73.1456757, 73.0310287
16: -48.6224174, 37.1973572, -48.3850174, 36.9610138, -85.5834351, 85.5823746
17: -79.7042236, 34.0010223, -79.2772064, 33.6349831, -113.3392029, 113.2782288
18: -48.2516060, 33.4623947, -48.0643959, 33.2125778, -81.4641876, 81.5267944
19: -38.2860641, 19.2800369, -38.1222763, 19.1796589, -57.4657211, 57.4023132
20: -34.7844391, 24.9553185, -34.6433754, 24.8322582, -59.6166992, 59.5986938
21: -46.2361946, 24.8947983, -46.0751915, 24.7737465, -71.0099411, 70.9699860
22: -49.1502571, 25.1926689, -48.9621391, 25.0215530, -74.1718140, 74.1548080
23: -37.8649483, 26.3669167, -37.7112236, 26.2613106, -64.1262589, 64.0781403
24: -45.4526443, 28.8574924, -45.2257996, 28.7786961, -74.2313385, 74.0832901
25: -39.7283707, 29.4394665, -39.5844078, 29.3347416, -69.0631104, 69.0238724
26: -56.0447540, 38.7844315, -55.9106407, 38.4928360, -94.5375900, 94.6950684
27: -46.0213127, 30.1588402, -45.7783585, 30.0313835, -76.0526962, 75.9371948
28: -36.9949608, 29.8552933, -36.8612709, 29.7418747, -66.7368317, 66.7165680
29: -51.2277298, 24.5889683, -51.0644150, 24.4374390, -75.6651688, 75.6533813
30: -46.3489113, 33.4442902, -46.2010803, 33.2505188, -79.5994263, 79.6453705
31: -49.1294174, 27.7798462, -48.9012756, 27.6534081, -76.7828217, 76.6811218
32: -55.5901413, 24.6363068, -55.4486961, 24.4048958, -79.8752136, 79.9588623
33: -73.7042770, 31.8676910, -73.4205933, 31.6559849, -104.7834244, 104.7036743
34: -63.7079735, 17.9690800, -63.5587730, 17.7605839, -81.0849915, 81.1529236
35: -60.7963028, 24.3963318, -60.6178207, 24.2788639, -84.4782867, 84.3974304
36: -60.8693008, 25.3412151, -60.7344971, 25.1990566, -86.0667725, 86.0744324
37: -89.4042130, 18.7516556, -89.0815735, 18.3499718, -107.5818253, 107.6658936
38: -69.7795715, 29.0676727, -69.5922852, 28.9294777, -98.7090454, 98.6599579
39: -83.3930969, 30.7763977, -83.1165848, 30.6182976, -114.0113983, 113.8929825
40: -65.7637939, 21.6668510, -65.4812241, 21.2563686, -86.9926376, 87.1177750
41: -58.7135277, 28.8183212, -58.5072632, 28.4649162, -87.1784439, 87.3255844
42: -40.1880493, 24.7204437, -40.0461388, 24.4261360, -64.6141815, 64.7665863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
time: 113.68 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
time: 103.29 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -57.2984505, 42.9293823, -57.0978622, 42.8727875, -100.1712341, 100.0272446
1: -26.5843811, 35.2697411, -26.4408741, 35.2285309, -61.8129120, 61.7106171
2: -24.4382973, 37.0292587, -24.2208538, 36.9904976, -61.4287949, 61.2501144
3: -28.7041702, 41.4008636, -28.5168934, 41.3356705, -70.0398407, 69.9177551
4: -31.6571388, 41.2758865, -31.4246922, 41.2149544, -72.8720932, 72.7005768
5: -28.0713692, 42.6871109, -27.8866405, 42.6226196, -70.6939850, 70.5737534
6: -55.2232437, 27.2871666, -55.1499138, 27.1111431, -82.3343887, 82.4370804
7: -32.5150146, 40.5239182, -32.3291245, 40.4716225, -72.9866333, 72.8530426
8: -37.2126007, 49.3988266, -36.9625626, 49.3336906, -86.5462952, 86.3613892
9: -30.3216248, 38.3344345, -30.2345200, 38.1603622, -68.4819870, 68.5689545
10: -49.6073456, 48.0996399, -49.4950066, 47.7698250, -97.3771667, 97.5946503
11: -48.5716820, 29.0418873, -48.4687958, 28.8260612, -77.3977432, 77.5106812
12: -59.8395767, 31.3727341, -59.7679977, 31.0028954, -90.4211273, 90.7165680
13: -51.4652061, 46.9701691, -51.2790146, 46.8181992, -98.2834015, 98.2491837
14: -79.7785187, 42.5672379, -79.5028915, 42.3098488, -122.0883636, 122.0701294
15: -38.1432800, 35.1911240, -37.9322624, 35.0936432, -73.2369232, 73.1233826
16: -48.6448708, 37.2030525, -48.5159912, 37.0067368, -85.6516113, 85.7190399
17: -79.7131653, 34.0608368, -79.5133972, 33.8438110, -113.5569763, 113.5742340
18: -48.2779427, 33.4693069, -48.1770592, 33.3266678, -81.6046143, 81.6463623
19: -38.3085785, 19.2827663, -38.2211723, 19.2348690, -57.5434494, 57.5039368
20: -34.7916527, 24.9716873, -34.7084999, 24.8940048, -59.6856575, 59.6801872
21: -46.2488670, 24.8990097, -46.1574440, 24.8104267, -71.0592957, 71.0564575
22: -49.1588020, 25.2020741, -49.0156364, 25.0989113, -74.2577133, 74.2177124
23: -37.8852043, 26.3719616, -37.8093796, 26.3106899, -64.1958923, 64.1813431
24: -45.4734268, 28.8608856, -45.3156319, 28.8201523, -74.2935791, 74.1765137
25: -39.7398987, 29.4466705, -39.6481361, 29.3896561, -69.1295547, 69.0948029
26: -56.0557404, 38.7939606, -55.9695930, 38.5859337, -94.6416779, 94.7635498
27: -46.0352745, 30.1645622, -45.8522530, 30.0761375, -76.1114120, 76.0168152
28: -37.0040359, 29.8684330, -36.9194946, 29.8064804, -66.8105164, 66.7879257
29: -51.2356796, 24.5947895, -51.1325073, 24.4893799, -75.7250595, 75.7272949
30: -46.3591537, 33.4565010, -46.2775879, 33.3232231, -79.6823730, 79.7340851
31: -49.1604424, 27.7836514, -49.0341644, 27.7250023, -76.8854446, 76.8178177
32: -55.6089973, 24.6456871, -55.5343475, 24.4806175, -79.9686890, 80.0520630
33: -73.7446289, 31.8746758, -73.5628510, 31.7809296, -104.9601288, 104.8458328
34: -63.7195129, 17.9750767, -63.6150894, 17.8355713, -81.1720123, 81.2065277
35: -60.8065186, 24.4004803, -60.6689110, 24.3385277, -84.5529175, 84.4483337
36: -60.8758011, 25.3469067, -60.7785950, 25.2566566, -86.1310806, 86.1243057
37: -89.4729996, 18.7578087, -89.3224716, 18.5634384, -107.8670502, 107.9104156
38: -69.7935028, 29.0773125, -69.6847839, 29.0045891, -98.7980957, 98.7621002
39: -83.4377441, 30.7820759, -83.2919006, 30.7383251, -114.1760712, 114.0739746
40: -65.8224106, 21.6737118, -65.6921539, 21.4613266, -87.2590027, 87.3337402
41: -58.7633057, 28.8244171, -58.6809502, 28.6181278, -87.3814316, 87.5053711
42: -40.2148209, 24.7286339, -40.1535263, 24.5287819, -64.7436066, 64.8821564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
time: 136.12 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
time: 109.11 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -57.3129311, 42.9134712, -57.1015968, 42.8685722, -100.1815033, 100.0150681
1: -26.5978661, 35.2501297, -26.4281712, 35.2185211, -61.8163872, 61.6782990
2: -24.4760303, 37.0081673, -24.2926941, 37.0374603, -61.5134888, 61.3008614
3: -28.7319412, 41.3685150, -28.5049553, 41.3340454, -70.0659866, 69.8734741
4: -31.6937771, 41.2570267, -31.5021305, 41.3031464, -72.9969254, 72.7591553
5: -28.0970936, 42.6632118, -27.8978710, 42.6444435, -70.7415390, 70.5610809
6: -55.2220612, 27.3028107, -55.2391090, 27.1552925, -82.3773499, 82.5419159
7: -32.5375595, 40.4949493, -32.3358154, 40.4308624, -72.9684219, 72.8307648
8: -37.2536964, 49.3620300, -36.9930840, 49.3518486, -86.6055450, 86.3551178
9: -30.3277168, 38.3795166, -30.3622608, 38.3254013, -68.6531219, 68.7417755
10: -49.6024399, 48.1977348, -49.8192062, 48.0489273, -97.6513672, 98.0169373
11: -48.5747375, 29.1061134, -48.7344017, 29.0409431, -77.6156769, 77.8405151
12: -59.8302116, 31.4635353, -60.0780144, 31.2624149, -90.6663742, 91.1254044
13: -51.4673233, 46.9706154, -51.2571411, 46.8814430, -98.3487701, 98.2277527
14: -79.7893677, 42.5832214, -79.6222000, 42.4013748, -122.1907425, 122.2054214
15: -38.1922684, 35.1964188, -38.1012650, 35.2523956, -73.4446640, 73.2976837
16: -48.6490250, 37.2593613, -48.7115746, 37.1976089, -85.8466339, 85.9709320
17: -79.7186127, 34.0760803, -79.5943451, 33.9422951, -113.6609039, 113.6704254
18: -48.2753220, 33.4793587, -48.2547417, 33.3146515, -81.5899734, 81.7341003
19: -38.3034439, 19.2867699, -38.2622757, 19.2280807, -57.5315247, 57.5490456
20: -34.8012772, 24.9764023, -34.7942810, 24.9181213, -59.7193985, 59.7706833
21: -46.2555046, 24.9158058, -46.3007393, 24.8705750, -71.1260834, 71.2165451
22: -49.1782150, 25.2122459, -49.1212349, 25.1787148, -74.3569336, 74.3334808
23: -37.8799820, 26.3723240, -37.8353348, 26.3091316, -64.1891174, 64.2076569
24: -45.4995575, 28.8644371, -45.4269409, 28.8756142, -74.3751678, 74.2913818
25: -39.7479324, 29.4529057, -39.6959915, 29.4313736, -69.1793060, 69.1488953
26: -56.0666122, 38.8230782, -56.0875893, 38.6710205, -94.7376328, 94.9106674
27: -46.0833244, 30.1654091, -46.0405273, 30.1727142, -76.2560425, 76.2059326
28: -37.0175705, 29.8643093, -36.9765358, 29.8463326, -66.8639069, 66.8408432
29: -51.2492294, 24.6115284, -51.2055435, 24.5523319, -75.8015594, 75.8170700
30: -46.3671074, 33.4721031, -46.3300552, 33.3816643, -79.7487717, 79.8021545
31: -49.1548271, 27.7895584, -49.0896912, 27.7118111, -76.8666382, 76.8792496
32: -55.6089630, 24.6781273, -55.6402588, 24.5688019, -80.0571823, 80.1962280
33: -73.7642899, 31.8835411, -73.6565552, 31.9179916, -105.1404572, 104.9645615
34: -63.7380180, 17.9817505, -63.6928596, 17.9503479, -81.3565216, 81.3142700
35: -60.8419991, 24.4086227, -60.7992210, 24.5214863, -84.8026428, 84.5979691
36: -60.8937607, 25.3515930, -60.8581047, 25.3222485, -86.2149963, 86.2086182
37: -89.4403839, 18.7659760, -89.2783585, 18.4686241, -107.7431488, 107.8805466
38: -69.8042526, 29.0816517, -69.7517929, 29.0554008, -98.8596497, 98.8334427
39: -83.4239502, 30.7854576, -83.2961273, 30.7712383, -114.1951904, 114.0815887
40: -65.8000336, 21.6712379, -65.6840897, 21.3245316, -87.1042252, 87.3303604
41: -58.7344894, 28.8270569, -58.6423836, 28.5277996, -87.2622910, 87.4694366
42: -40.2047424, 24.7571640, -40.2864380, 24.5894775, -64.7942200, 65.0436020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523
time: 75.78 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523
time: 127.30 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -57.3274307, 42.9422989, -57.2684784, 42.9679871, -100.2954178, 100.2107773
1: -26.6024132, 35.2786751, -26.5484638, 35.3180618, -61.9204750, 61.8271408
2: -24.4816818, 37.0371132, -24.4037437, 37.1371117, -61.6187935, 61.4408569
3: -28.7376213, 41.4166565, -28.6664162, 41.4946022, -70.2322235, 70.0830688
4: -31.6996975, 41.2880707, -31.6203823, 41.4160995, -73.1157990, 72.9084549
5: -28.1030769, 42.7030678, -28.0374603, 42.7798538, -70.8829346, 70.7405243
6: -55.2433472, 27.3110962, -55.3303337, 27.2355804, -82.4789276, 82.6414337
7: -32.5451927, 40.5355110, -32.5036926, 40.5640945, -73.1092834, 73.0391998
8: -37.2604790, 49.4131660, -37.1669159, 49.5306587, -86.7911377, 86.5800781
9: -30.3364182, 38.3867874, -30.4293575, 38.3736572, -68.7100754, 68.8161469
10: -49.6301727, 48.2063179, -49.9513779, 48.1635513, -97.7937241, 98.1576996
11: -48.5925789, 29.1109867, -48.8616562, 29.0783272, -77.6709061, 77.9726410
12: -59.8560028, 31.4784412, -60.1709290, 31.4033508, -90.8368301, 91.2295456
13: -51.4777222, 47.0191650, -51.3962250, 47.0546379, -98.5323639, 98.4153900
14: -79.8015747, 42.6628189, -79.8701401, 42.6549339, -122.4565125, 122.5329590
15: -38.2020950, 35.2096481, -38.1804924, 35.3337173, -73.5358124, 73.3901367
16: -48.6715622, 37.2650452, -48.8424149, 37.2434464, -85.9150085, 86.1074600
17: -79.7275848, 34.1362000, -79.8305817, 34.1516876, -113.8792725, 113.9667816
18: -48.3016434, 33.4862633, -48.3676147, 33.4289246, -81.7305679, 81.8538818
19: -38.3260727, 19.2895031, -38.3613586, 19.2832661, -57.6093369, 57.6508636
20: -34.8085098, 24.9927845, -34.8593979, 24.9797974, -59.7883072, 59.8521805
21: -46.2681503, 24.9200211, -46.3828163, 24.9073124, -71.1754608, 71.3028412
22: -49.1867561, 25.2215958, -49.1748505, 25.2561378, -74.4428940, 74.3964462
23: -37.9002533, 26.3773632, -37.9334488, 26.3585091, -64.2587585, 64.3108139
24: -45.5203667, 28.8678207, -45.5169983, 28.9170723, -74.4374390, 74.3848190
25: -39.7594681, 29.4600983, -39.7598801, 29.4863930, -69.2458649, 69.2199783
26: -56.0775414, 38.8326492, -56.1465034, 38.7641296, -94.8416748, 94.9791565
27: -46.0973129, 30.1711044, -46.1146774, 30.2174721, -76.3147888, 76.2857819
28: -37.0266418, 29.8774529, -37.0349045, 29.9109726, -66.9376144, 66.9123535
29: -51.2571640, 24.6173706, -51.2736893, 24.6043625, -75.8615265, 75.8910599
30: -46.3773117, 33.4843445, -46.4070854, 33.4542198, -79.8315277, 79.8914337
31: -49.1858559, 27.7933235, -49.2227249, 27.7834587, -76.9693146, 77.0160522
32: -55.6278572, 24.6875362, -55.7258987, 24.6447086, -80.1508484, 80.2893372
33: -73.8046188, 31.8905602, -73.7987976, 32.0429497, -105.3171082, 105.1067810
34: -63.7495956, 17.9877281, -63.7491188, 18.0252762, -81.4434738, 81.3680420
35: -60.8521957, 24.4127178, -60.8503914, 24.5811253, -84.8771896, 84.6491165
36: -60.9002724, 25.3572044, -60.9023018, 25.3798504, -86.2792816, 86.2585220
37: -89.5092239, 18.7720966, -89.5192871, 18.6821880, -108.0283966, 108.1250153
38: -69.8181686, 29.0912609, -69.8444214, 29.1305828, -98.9487534, 98.9356842
39: -83.4685745, 30.7911682, -83.4716187, 30.8912315, -114.3598022, 114.2627869
40: -65.8586731, 21.6780968, -65.8950806, 21.5295410, -87.3707275, 87.5464478
41: -58.7842751, 28.8331585, -58.8161812, 28.6810646, -87.4653397, 87.6493378
42: -40.2315445, 24.7653618, -40.3937531, 24.6923027, -64.9238434, 65.1591187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=248, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=505, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523
time: 78.11 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523
time: 92.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 173.09 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1482238, upper bound: 38.0990002
IS_A1_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1482238, upper bound: 38.0990002
IS_A1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1753361, upper bound: 38.0990002
IS_A1_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1482238, upper bound: 38.0990002
IS_A1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1350350
IS_A1_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1350350
IS_A1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1630668
IS_A1_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1753362, upper bound: 38.1630668
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
IS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
IS_A2_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1482238, upper bound: 38.1448021
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 173.09
Output dim: 2, lower bound: -38.1483465, upper bound: 38.2080523

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -57.1942902, 42.8516541, -57.0640564, 42.8538628, -100.0481567, 99.9157104
1: -26.4995365, 35.1897125, -26.3980446, 35.2125320, -61.7120667, 61.5877571
2: -24.3756580, 36.9413071, -24.2587166, 37.0315742, -61.4072342, 61.2000237
3: -28.6418934, 41.3146172, -28.4751339, 41.3273544, -69.9692459, 69.7897491
4: -31.5585670, 41.1839752, -31.4563847, 41.2970848, -72.8556519, 72.6403580
5: -28.0148125, 42.5989113, -27.8711548, 42.6383133, -70.6531219, 70.4700623
6: -55.1536636, 27.2044106, -55.2287598, 27.1217690, -82.2754364, 82.4331665
7: -32.4537048, 40.4484329, -32.3114624, 40.4256287, -72.8793335, 72.7598953
8: -37.0507965, 49.2378998, -36.9224739, 49.3405495, -86.3913422, 86.1603699
9: -30.2414379, 38.3254929, -30.3335514, 38.3183556, -68.5597916, 68.6590424
10: -49.5159416, 48.1305237, -49.7936134, 48.0355377, -97.5514832, 97.9241333
11: -48.4206009, 28.9662151, -48.7189713, 28.9927769, -77.4133759, 77.6851883
12: -59.7640686, 31.3518009, -60.0697403, 31.2260971, -90.5641479, 91.0053253
13: -51.3401146, 46.8619804, -51.2154541, 46.8632736, -98.2033844, 98.0774384
14: -79.6768188, 42.5452423, -79.5857239, 42.3855324, -122.0623474, 122.1309662
15: -38.0552483, 35.1260376, -38.0557327, 35.2440987, -73.2993469, 73.1817703
16: -48.5751266, 37.1970367, -48.6965294, 37.1774521, -85.7525787, 85.8935699
17: -79.5716705, 34.0137100, -79.5734329, 33.9221840, -113.4938507, 113.5871429
18: -48.1570587, 33.3117485, -48.2397385, 33.2560692, -81.4131317, 81.5514832
19: -38.2196045, 19.2250843, -38.2508583, 19.2058392, -57.4254456, 57.4759445
20: -34.7475128, 24.9008255, -34.7820206, 24.8926830, -59.6401978, 59.6828461
21: -46.1417961, 24.8217564, -46.2860832, 24.8372879, -70.9790802, 71.1078415
22: -49.0853043, 25.1429729, -49.1060104, 25.1558323, -74.2411346, 74.2489853
23: -37.8015633, 26.2724705, -37.8253975, 26.2756252, -64.0771866, 64.0978699
24: -45.4148483, 28.7629051, -45.4143486, 28.8409882, -74.2558365, 74.1772537
25: -39.6910858, 29.3799801, -39.6849136, 29.4079857, -69.0990753, 69.0648956
26: -55.9539948, 38.7090683, -56.0715790, 38.6302490, -94.5842438, 94.7806473
27: -45.9991226, 30.0959473, -46.0244827, 30.1491909, -76.1483154, 76.1204300
28: -36.9352303, 29.7531967, -36.9664001, 29.8083878, -66.7436218, 66.7195969
29: -51.1365356, 24.5317726, -51.1898727, 24.5251045, -75.6616364, 75.7216492
30: -46.2499542, 33.3130646, -46.3157845, 33.3275070, -79.5774612, 79.6288452
31: -49.0614357, 27.6830978, -49.0752602, 27.6750622, -76.7364960, 76.7583618
32: -55.5539856, 24.5867500, -55.6310768, 24.5405884, -79.9744110, 80.0957870
33: -73.7191696, 31.8172398, -73.6446457, 31.8999901, -105.0865784, 104.8932877
34: -63.6512680, 17.8658638, -63.6820831, 17.9155331, -81.2362061, 81.1855316
35: -60.7711868, 24.3155861, -60.7890511, 24.4941311, -84.7050018, 84.4946136
36: -60.8410110, 25.2861786, -60.8511848, 25.3021164, -86.1421356, 86.1362152
37: -89.3533173, 18.6723118, -89.2662659, 18.4385490, -107.6260834, 107.7743530
38: -69.7422867, 29.0317955, -69.7397079, 29.0445271, -98.7868118, 98.7714996
39: -83.3531799, 30.7125072, -83.2767029, 30.7640800, -114.1172638, 113.9892120
40: -65.7398529, 21.5965271, -65.6701202, 21.3043022, -87.0260849, 87.2420654
41: -58.6687584, 28.7379341, -58.6319504, 28.4980106, -87.1667709, 87.3698883
42: -40.1692734, 24.7059975, -40.2771568, 24.5742702, -64.7435455, 64.9831543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
time: 88.55 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
time: 93.82 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -57.3051758, 42.9107971, -57.0991058, 42.8677979, -100.1729736, 100.0099030
1: -26.5947819, 35.2488441, -26.4273071, 35.2181816, -61.8129654, 61.6761513
2: -24.4726906, 37.0070496, -24.2917404, 37.0371704, -61.5098610, 61.2987900
3: -28.7289867, 41.3673630, -28.5041561, 41.3337212, -70.0627060, 69.8715210
4: -31.6893768, 41.2557602, -31.5008755, 41.3027954, -72.9921722, 72.7566376
5: -28.0943680, 42.6622467, -27.8971004, 42.6441994, -70.7385712, 70.5593491
6: -55.2201767, 27.2989044, -55.2385635, 27.1539516, -82.3741302, 82.5374680
7: -32.5346451, 40.4942093, -32.3349648, 40.4306526, -72.9653015, 72.8291779
8: -37.2471008, 49.3601990, -36.9913101, 49.3513489, -86.5984497, 86.3515091
9: -30.3247871, 38.3784103, -30.3614559, 38.3250771, -68.6498642, 68.7398682
10: -49.5996323, 48.1957169, -49.8184357, 48.0483665, -97.6479950, 98.0141525
11: -48.5720711, 29.1017380, -48.7336159, 29.0397453, -77.6118164, 77.8353577
12: -59.8290787, 31.4598389, -60.0777283, 31.2614021, -90.6646881, 91.1104965
13: -51.4632492, 46.9682312, -51.2559853, 46.8807564, -98.3440094, 98.2242126
14: -79.7777786, 42.5811234, -79.6189880, 42.4007759, -122.1785583, 122.2001114
15: -38.1880417, 35.1948242, -38.1001358, 35.2519073, -73.4399490, 73.2949600
16: -48.6462059, 37.2558670, -48.7107735, 37.1966400, -85.8428497, 85.9666443
17: -79.7153702, 34.0733299, -79.5934448, 33.9415283, -113.6568985, 113.6667786
18: -48.2733498, 33.4739418, -48.2541656, 33.3131790, -81.5865326, 81.7281036
19: -38.3014145, 19.2846260, -38.2616959, 19.2274895, -57.5289040, 57.5463219
20: -34.7990723, 24.9674187, -34.7936554, 24.9154301, -59.7145004, 59.7610741
21: -46.2530327, 24.9127464, -46.3000298, 24.8697395, -71.1227722, 71.2127762
22: -49.1757698, 25.2098866, -49.1205597, 25.1780357, -74.3538055, 74.3304443
23: -37.8783264, 26.3690376, -37.8348694, 26.3082447, -64.1865692, 64.2039032
24: -45.4978371, 28.8608475, -45.4264526, 28.8746490, -74.3724823, 74.2873001
25: -39.7463684, 29.4503021, -39.6955719, 29.4306126, -69.1769791, 69.1458740
26: -56.0639572, 38.8191376, -56.0868530, 38.6698914, -94.7338486, 94.9059906
27: -46.0808105, 30.1632156, -46.0398102, 30.1721230, -76.2529297, 76.2030258
28: -37.0156021, 29.8608475, -36.9760132, 29.8453484, -66.8609467, 66.8368607
29: -51.2463913, 24.6087990, -51.2047424, 24.5515556, -75.7979431, 75.8135376
30: -46.3648300, 33.4670868, -46.3293915, 33.3802872, -79.7451172, 79.7964783
31: -49.1524429, 27.7858982, -49.0890236, 27.7108212, -76.8632660, 76.8749237
32: -55.6072235, 24.6753750, -55.6397743, 24.5680771, -80.0548172, 80.1876678
33: -73.7629013, 31.8775139, -73.6561584, 31.9157104, -105.1342392, 104.9781647
34: -63.7367477, 17.9781094, -63.6925011, 17.9493332, -81.3554230, 81.2761230
35: -60.8407211, 24.4055138, -60.7988815, 24.5206261, -84.8014526, 84.5712280
36: -60.8927078, 25.3494759, -60.8578033, 25.3217010, -86.2134247, 86.2057877
37: -89.4384918, 18.7626362, -89.2778168, 18.4677238, -107.7407455, 107.8662949
38: -69.8025208, 29.0799732, -69.7512894, 29.0549259, -98.8574448, 98.8312607
39: -83.4216919, 30.7839851, -83.2955322, 30.7707844, -114.1924744, 114.0795135
40: -65.7975235, 21.6641922, -65.6833725, 21.3225403, -87.0999603, 87.3201752
41: -58.7329674, 28.8240662, -58.6419678, 28.5269604, -87.2599258, 87.4660339
42: -40.2034607, 24.7516747, -40.2860870, 24.5879097, -64.7913666, 65.0377655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
time: 85.23 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
time: 87.13 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -57.2088013, 42.8804932, -57.2308540, 42.9532890, -100.1620941, 100.1113434
1: -26.5040894, 35.2182693, -26.5183640, 35.3120575, -61.8161469, 61.7366333
2: -24.3812923, 36.9702454, -24.3697510, 37.1312027, -61.5124969, 61.3399963
3: -28.6475983, 41.3627777, -28.6366024, 41.4879150, -70.1355133, 69.9993820
4: -31.5644722, 41.2149963, -31.5746307, 41.4100418, -72.9745178, 72.7896271
5: -28.0208054, 42.6387787, -28.0107498, 42.7736778, -70.7944794, 70.6495285
6: -55.1749153, 27.2127495, -55.3199539, 27.2020683, -82.3769836, 82.5326996
7: -32.4613419, 40.4889679, -32.4793472, 40.5588837, -73.0202255, 72.9683151
8: -37.0576019, 49.2890091, -37.0962830, 49.5193596, -86.5769653, 86.3852921
9: -30.2501450, 38.3327866, -30.4006672, 38.3666077, -68.6167526, 68.7334518
10: -49.5437164, 48.1391258, -49.9257889, 48.1501732, -97.6938934, 98.0649109
11: -48.4384613, 28.9710941, -48.8462639, 29.0301857, -77.4686432, 77.8173599
12: -59.7898636, 31.3667412, -60.1626434, 31.3670540, -90.7345963, 91.1094971
13: -51.3505325, 46.9105377, -51.3545914, 47.0364609, -98.3869934, 98.2651291
14: -79.6890106, 42.6248016, -79.8336563, 42.6390991, -122.3281097, 122.4584579
15: -38.0650749, 35.1392822, -38.1349792, 35.3254166, -73.3904877, 73.2742615
16: -48.5976486, 37.2027550, -48.8273621, 37.2232666, -85.8209152, 86.0301208
17: -79.5806122, 34.0737610, -79.8096924, 34.1316032, -113.7122192, 113.8834534
18: -48.1833992, 33.3186531, -48.3526001, 33.3703117, -81.5537109, 81.6712494
19: -38.2422371, 19.2278290, -38.3499298, 19.2610416, -57.5032806, 57.5777588
20: -34.7547607, 24.9172173, -34.8471565, 24.9543610, -59.7091217, 59.7643738
21: -46.1544418, 24.8259754, -46.3681526, 24.8740273, -71.0284729, 71.1941299
22: -49.0938454, 25.1523705, -49.1595879, 25.2332458, -74.3270874, 74.3119583
23: -37.8218384, 26.2775269, -37.9235153, 26.3250027, -64.1468430, 64.2010422
24: -45.4356384, 28.7663155, -45.5044060, 28.8824310, -74.3180695, 74.2707214
25: -39.7026291, 29.3871613, -39.7488022, 29.4629765, -69.1656036, 69.1359634
26: -55.9649544, 38.7186356, -56.1304855, 38.7233582, -94.6883087, 94.8491211
27: -46.0131264, 30.1016731, -46.0986214, 30.1939144, -76.2070389, 76.2002945
28: -36.9443054, 29.7663364, -37.0247803, 29.8730621, -66.8173676, 66.7911148
29: -51.1444702, 24.5376320, -51.2580414, 24.5771523, -75.7216187, 75.7956696
30: -46.2601852, 33.3253326, -46.3928413, 33.4001083, -79.6602936, 79.7181702
31: -49.0924530, 27.6868896, -49.2082939, 27.7466946, -76.8391495, 76.8951874
32: -55.5728912, 24.5961800, -55.7167206, 24.6164932, -80.0680618, 80.1889114
33: -73.7595139, 31.8242435, -73.7868958, 32.0249405, -105.2631912, 105.0355148
34: -63.6628342, 17.8718185, -63.7383156, 17.9904881, -81.3231506, 81.2392654
35: -60.7813988, 24.3197212, -60.8401756, 24.5537586, -84.7796021, 84.5457001
36: -60.8475380, 25.2918129, -60.8953705, 25.3597488, -86.2063904, 86.1861267
37: -89.4221115, 18.6784515, -89.5072250, 18.6520710, -107.9113541, 108.0188065
38: -69.7562332, 29.0414200, -69.8323059, 29.1197243, -98.8759613, 98.8737259
39: -83.3978195, 30.7182293, -83.4521637, 30.8841038, -114.2819214, 114.1703949
40: -65.7985001, 21.6034050, -65.8811340, 21.5093269, -87.2925797, 87.4581451
41: -58.7185287, 28.7440414, -58.8057060, 28.6512871, -87.3698120, 87.5497437
42: -40.1960640, 24.7142162, -40.3844681, 24.6771069, -64.8731689, 65.0986862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1300659, upper bound: 38.2080522
time: 146.65 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
time: 72.52 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -57.3196907, 42.9396133, -57.2659721, 42.9672394, -100.2869263, 100.2055817
1: -26.5993462, 35.2774277, -26.5476093, 35.3177109, -61.9170570, 61.8250351
2: -24.4783306, 37.0360107, -24.4028072, 37.1367874, -61.6151199, 61.4388199
3: -28.7346649, 41.4155121, -28.6656132, 41.4942894, -70.2289581, 70.0811234
4: -31.6953049, 41.2867889, -31.6191483, 41.4157333, -73.1110382, 72.9059372
5: -28.1003609, 42.7021179, -28.0367012, 42.7795563, -70.8799133, 70.7388153
6: -55.2414398, 27.3072243, -55.3297958, 27.2342339, -82.4756775, 82.6370239
7: -32.5423164, 40.5347595, -32.5028496, 40.5638885, -73.1062012, 73.0376129
8: -37.2538948, 49.4113426, -37.1651077, 49.5301590, -86.7840576, 86.5764465
9: -30.3334942, 38.3857040, -30.4285526, 38.3733139, -68.7068100, 68.8142548
10: -49.6273727, 48.2043343, -49.9505997, 48.1629944, -97.7903671, 98.1549377
11: -48.5899124, 29.1065941, -48.8608818, 29.0771408, -77.6670532, 77.9674759
12: -59.8548584, 31.4747467, -60.1706161, 31.4023495, -90.8351517, 91.2146835
13: -51.4736176, 47.0167542, -51.3951111, 47.0539513, -98.5275726, 98.4118652
14: -79.7899628, 42.6606903, -79.8669434, 42.6542740, -122.4442368, 122.5276337
15: -38.1978531, 35.2080574, -38.1793289, 35.3332253, -73.5310822, 73.3873901
16: -48.6687431, 37.2615662, -48.8416176, 37.2424698, -85.9112091, 86.1031799
17: -79.7243652, 34.1334152, -79.8296890, 34.1508904, -113.8752594, 113.9631042
18: -48.2996826, 33.4808655, -48.3670006, 33.4274368, -81.7271194, 81.8478699
19: -38.3240509, 19.2873592, -38.3607712, 19.2826748, -57.6067276, 57.6481323
20: -34.8062897, 24.9838009, -34.8587723, 24.9771061, -59.7833939, 59.8425751
21: -46.2656975, 24.9169559, -46.3821030, 24.9064674, -71.1721649, 71.2990570
22: -49.1842957, 25.2192383, -49.1741638, 25.2554436, -74.4397430, 74.3934021
23: -37.8986053, 26.3740768, -37.9329987, 26.3576069, -64.2562103, 64.3070755
24: -45.5186539, 28.8642426, -45.5165176, 28.9160919, -74.4347458, 74.3807602
25: -39.7579002, 29.4574966, -39.7594528, 29.4856110, -69.2435150, 69.2169495
26: -56.0749359, 38.8286552, -56.1457367, 38.7630463, -94.8379822, 94.9743958
27: -46.0947762, 30.1688824, -46.1139336, 30.2168484, -76.3116226, 76.2828140
28: -37.0246658, 29.8739738, -37.0343590, 29.9100208, -66.9346848, 66.9083328
29: -51.2543144, 24.6146393, -51.2728844, 24.6035919, -75.8579102, 75.8875275
30: -46.3750648, 33.4793167, -46.4064522, 33.4528351, -79.8278961, 79.8857727
31: -49.1834869, 27.7896729, -49.2220573, 27.7824516, -76.9659424, 77.0117340
32: -55.6260948, 24.6847687, -55.7253952, 24.6439552, -80.1484680, 80.2808075
33: -73.8032227, 31.8845711, -73.7983856, 32.0406723, -105.3108978, 105.1203842
34: -63.7483139, 17.9840508, -63.7487411, 18.0242691, -81.4423676, 81.3299179
35: -60.8509331, 24.4096355, -60.8500137, 24.5802593, -84.8760300, 84.6223679
36: -60.8992195, 25.3551331, -60.9020119, 25.3792801, -86.2776794, 86.2557220
37: -89.5073090, 18.7688217, -89.5187531, 18.6812916, -108.0260010, 108.1107635
38: -69.8164368, 29.0895863, -69.8438950, 29.1301250, -98.9465637, 98.9334793
39: -83.4663467, 30.7896748, -83.4709702, 30.8908234, -114.3571701, 114.2606430
40: -65.8561783, 21.6710930, -65.8943634, 21.5275459, -87.3664551, 87.5362854
41: -58.7827263, 28.8301468, -58.8157501, 28.6802406, -87.4629669, 87.6458969
42: -40.2302704, 24.7598667, -40.3933945, 24.6907234, -64.9209900, 65.1532593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=247, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1721

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
time: 88.60 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080523
time: 78.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 170.06 seconds
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1300659, upper bound: 38.2080522
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 170.06
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080523

## BFS IS instance: IS_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -57.1942902, 42.8516541, -57.0231323, 42.7731018, -99.9673920, 99.8747864
1: -26.4995365, 35.1897125, -26.3849812, 35.1355515, -61.6350861, 61.5746918
2: -24.3756580, 36.9413071, -24.2416534, 36.9339180, -61.3095779, 61.1829605
3: -28.6418934, 41.3146172, -28.4588490, 41.2496567, -69.8915482, 69.7734680
4: -31.5585670, 41.1839752, -31.4405689, 41.2015152, -72.7600861, 72.6245422
5: -28.0148125, 42.5989113, -27.8539104, 42.5463486, -70.5611572, 70.4528198
6: -55.1536636, 27.2044106, -55.1421432, 27.0972404, -82.2509003, 82.3465576
7: -32.4537048, 40.4484329, -32.2913132, 40.3327103, -72.7864151, 72.7397461
8: -37.0507965, 49.2378998, -36.9049454, 49.2197113, -86.2705078, 86.1428452
9: -30.2414379, 38.3254929, -30.2931290, 38.2963562, -68.5377960, 68.6186218
10: -49.5159416, 48.1305237, -49.7341461, 48.0076408, -97.5235825, 97.8646698
11: -48.4206009, 28.9662151, -48.6553726, 28.9668961, -77.3874969, 77.6215897
12: -59.7640686, 31.3518009, -59.9752045, 31.1889572, -90.5301208, 90.9119263
13: -51.3401146, 46.8619804, -51.1816330, 46.7419395, -98.0820541, 98.0436096
14: -79.6768188, 42.5452423, -79.5461502, 42.1637993, -121.8406219, 122.0913925
15: -38.0552483, 35.1260376, -38.0216446, 35.2068062, -73.2620544, 73.1476822
16: -48.5751266, 37.1970367, -48.6303787, 37.1544876, -85.7296143, 85.8274155
17: -79.5716705, 34.0137100, -79.5412750, 33.7642937, -113.3359680, 113.5549850
18: -48.1570587, 33.3117485, -48.1837807, 33.2336426, -81.3907013, 81.4955292
19: -38.2196045, 19.2250843, -38.2143631, 19.1920586, -57.4116631, 57.4394455
20: -34.7475128, 24.9008255, -34.7623329, 24.8491745, -59.5966873, 59.6631584
21: -46.1417961, 24.8217564, -46.2537308, 24.8182373, -70.9600372, 71.0754852
22: -49.0853043, 25.1429729, -49.0758476, 25.1284237, -74.2137299, 74.2188187
23: -37.8015633, 26.2724705, -37.7798767, 26.2515259, -64.0530853, 64.0523453
24: -45.4148483, 28.7629051, -45.3814354, 28.8157616, -74.2306061, 74.1443405
25: -39.6910858, 29.3799801, -39.6559372, 29.3588200, -69.0499039, 69.0359192
26: -55.9539948, 38.7090683, -55.9987717, 38.6012421, -94.5552368, 94.7078400
27: -45.9991226, 30.0959473, -45.9779739, 30.1317844, -76.1309052, 76.0739212
28: -36.9352303, 29.7531967, -36.9303284, 29.7800217, -66.7152557, 66.6835251
29: -51.1365356, 24.5317726, -51.1601868, 24.5086899, -75.6452255, 75.6919556
30: -46.2499542, 33.3130646, -46.2864990, 33.2967072, -79.5466614, 79.5995636
31: -49.0614357, 27.6830978, -49.0206528, 27.6517010, -76.7131348, 76.7037506
32: -55.5539856, 24.5867500, -55.5589142, 24.5154724, -79.9485474, 80.0240631
33: -73.7191696, 31.8172398, -73.5699387, 31.8809414, -105.0727997, 104.8199463
34: -63.6512680, 17.8658638, -63.5884857, 17.9001198, -81.2379456, 81.0813751
35: -60.7711868, 24.3155861, -60.7499046, 24.4858036, -84.6911621, 84.4537659
36: -60.8410110, 25.2861786, -60.7979050, 25.2858543, -86.1260376, 86.0828552
37: -89.3533173, 18.6723118, -89.0933914, 18.4200001, -107.6117859, 107.5974426
38: -69.7422867, 29.0317955, -69.7053375, 29.0150433, -98.7573318, 98.7371368
39: -83.3531799, 30.7125072, -83.2136154, 30.7451649, -114.0983429, 113.9261246
40: -65.7398529, 21.5965271, -65.4895325, 21.2825794, -87.0067444, 87.0655365
41: -58.6687584, 28.7379341, -58.4693565, 28.4761810, -87.1449432, 87.2072906
42: -40.1692734, 24.7059975, -40.1926575, 24.5508156, -64.7200928, 64.8986511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2072686
time: 81.13 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2072686
time: 89.33 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -57.1942902, 42.8516541, -57.1774673, 42.8660774, -100.0603638, 100.0291214
1: -26.4995365, 35.1897125, -26.4862080, 35.2251663, -61.7247009, 61.6759186
2: -24.3756580, 36.9413071, -24.3623867, 37.0424957, -61.4181519, 61.3036957
3: -28.6418934, 41.3146172, -28.5623569, 41.3387413, -69.9806366, 69.8769760
4: -31.5585670, 41.1839752, -31.5564365, 41.3165627, -72.8751297, 72.7404099
5: -28.0148125, 42.5989113, -27.9607468, 42.6515846, -70.6663971, 70.5596619
6: -55.1536636, 27.2044106, -55.2328300, 27.2101059, -82.3637695, 82.4372406
7: -32.4537048, 40.4484329, -32.4147148, 40.4386444, -72.8923492, 72.8631439
8: -37.0507965, 49.2378998, -37.0473328, 49.3589706, -86.4097672, 86.2852325
9: -30.2414379, 38.3254929, -30.3721390, 38.3574257, -68.5988617, 68.6976318
10: -49.5159416, 48.1305237, -49.8302002, 48.0951424, -97.6110840, 97.9607239
11: -48.4206009, 28.9662151, -48.7543793, 29.0330162, -77.4536133, 77.7205963
12: -59.7640686, 31.3518009, -60.0855865, 31.3285980, -90.6579056, 91.0105896
13: -51.3401146, 46.8619804, -51.3574829, 46.8830185, -98.2231293, 98.2194672
14: -79.6768188, 42.5452423, -79.7893066, 42.3950157, -122.0718384, 122.3345490
15: -38.0552483, 35.1260376, -38.1155891, 35.2822762, -73.3375244, 73.2416229
16: -48.5751266, 37.1970367, -48.7311249, 37.2143173, -85.7894440, 85.9281616
17: -79.5716705, 34.0137100, -79.7264938, 33.9459000, -113.5175705, 113.7402039
18: -48.1570587, 33.3117485, -48.2587471, 33.3410187, -81.4980774, 81.5704956
19: -38.2196045, 19.2250843, -38.2752304, 19.2308731, -57.4504776, 57.5003128
20: -34.7475128, 24.9008255, -34.8184814, 24.9086399, -59.6561508, 59.7193069
21: -46.1417961, 24.8217564, -46.3177605, 24.8646622, -71.0064545, 71.1395187
22: -49.0853043, 25.1429729, -49.1422157, 25.1986752, -74.2839813, 74.2851868
23: -37.8015633, 26.2724705, -37.8562546, 26.3066635, -64.1082306, 64.1287231
24: -45.4148483, 28.7629051, -45.4473381, 28.8581543, -74.2730026, 74.2102432
25: -39.6910858, 29.3799801, -39.7149429, 29.4220161, -69.1130981, 69.0949249
26: -55.9539948, 38.7090683, -56.0937920, 38.7212067, -94.6752014, 94.8028564
27: -45.9991226, 30.0959473, -46.0476608, 30.2129631, -76.2120819, 76.1436081
28: -36.9352303, 29.7531967, -36.9871063, 29.8348007, -66.7700348, 66.7403030
29: -51.1365356, 24.5317726, -51.2217560, 24.5672379, -75.7037735, 75.7535248
30: -46.2499542, 33.3130646, -46.3433380, 33.3752174, -79.6251678, 79.6564026
31: -49.0614357, 27.6830978, -49.1090240, 27.6991615, -76.7605972, 76.7921219
32: -55.5539856, 24.5867500, -55.6456337, 24.5966606, -80.0246277, 80.1087494
33: -73.7191696, 31.8172398, -73.6654129, 31.9497890, -105.1271286, 104.8986359
34: -63.6512680, 17.8658638, -63.7020302, 18.0139828, -81.3210297, 81.1579437
35: -60.7711868, 24.3155861, -60.8057251, 24.5221882, -84.7246094, 84.5085602
36: -60.8410110, 25.2861786, -60.8787498, 25.3603477, -86.2002029, 86.1633759
37: -89.3533173, 18.6723118, -89.2993698, 18.5929527, -107.7766495, 107.7932053
38: -69.7422867, 29.0317955, -69.7739182, 29.0641632, -98.8064499, 98.8057098
39: -83.3531799, 30.7125072, -83.3167725, 30.7805729, -114.1337509, 114.0292816
40: -65.7398529, 21.5965271, -65.6965942, 21.4917183, -87.2079544, 87.2621155
41: -58.6687584, 28.7379341, -58.6488953, 28.6679459, -87.3367004, 87.3868256
42: -40.1692734, 24.7059975, -40.2842369, 24.6577034, -64.8269806, 64.9902344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2072688
time: 107.60 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2075009
time: 84.99 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -57.3051758, 42.9107971, -57.0582275, 42.7870827, -100.0922546, 99.9690247
1: -26.5947819, 35.2488441, -26.4142208, 35.1412010, -61.7359848, 61.6630630
2: -24.4726906, 37.0070496, -24.2746887, 36.9394989, -61.4121895, 61.2817383
3: -28.7289867, 41.3673630, -28.4878483, 41.2559967, -69.9849854, 69.8552094
4: -31.6893768, 41.2557602, -31.4851112, 41.2072144, -72.8965912, 72.7408752
5: -28.0943680, 42.6622467, -27.8798809, 42.5522385, -70.6466064, 70.5421295
6: -55.2201767, 27.2989044, -55.1519623, 27.1294441, -82.3496246, 82.4508667
7: -32.5346451, 40.4942093, -32.3148232, 40.3377457, -72.8723907, 72.8090363
8: -37.2471008, 49.3601990, -36.9737892, 49.2305069, -86.4776077, 86.3339844
9: -30.3247871, 38.3784103, -30.3210640, 38.3030548, -68.6278381, 68.6994781
10: -49.5996323, 48.1957169, -49.7589989, 48.0204811, -97.6201172, 97.9547119
11: -48.5720711, 29.1017380, -48.6699982, 29.0138779, -77.5859528, 77.7717361
12: -59.8290787, 31.4598389, -59.9831772, 31.2243004, -90.6306458, 91.0170670
13: -51.4632492, 46.9682312, -51.2221451, 46.7594604, -98.2227097, 98.1903763
14: -79.7777786, 42.5811234, -79.5794678, 42.1790237, -121.9568024, 122.1605911
15: -38.1880417, 35.1948242, -38.0660362, 35.2146339, -73.4026794, 73.2608643
16: -48.6462059, 37.2558670, -48.6446342, 37.1736755, -85.8198853, 85.9004974
17: -79.7153702, 34.0733299, -79.5611954, 33.7836075, -113.4989777, 113.6345215
18: -48.2733498, 33.4739418, -48.1982155, 33.2907982, -81.5641479, 81.6721573
19: -38.3014145, 19.2846260, -38.2252045, 19.2136955, -57.5151100, 57.5098305
20: -34.7990723, 24.9674187, -34.7739601, 24.8719196, -59.6709900, 59.7413788
21: -46.2530327, 24.9127464, -46.2676544, 24.8507118, -71.1037445, 71.1804047
22: -49.1757698, 25.2098866, -49.0903854, 25.1506577, -74.3264313, 74.3002701
23: -37.8783264, 26.3690376, -37.7893181, 26.2841415, -64.1624680, 64.1583557
24: -45.4978371, 28.8608475, -45.3935318, 28.8494225, -74.3472595, 74.2543793
25: -39.7463684, 29.4503021, -39.6665802, 29.3814392, -69.1278076, 69.1168823
26: -56.0639572, 38.8191376, -56.0140114, 38.6408882, -94.7048492, 94.8331451
27: -46.0808105, 30.1632156, -45.9932747, 30.1546974, -76.2355042, 76.1564941
28: -37.0156021, 29.8608475, -36.9399376, 29.8170071, -66.8326111, 66.8007812
29: -51.2463913, 24.6087990, -51.1750336, 24.5351257, -75.7815170, 75.7838287
30: -46.3648300, 33.4670868, -46.3000946, 33.3494949, -79.7143250, 79.7671814
31: -49.1524429, 27.7858982, -49.0343933, 27.6874619, -76.8399048, 76.8202896
32: -55.6072235, 24.6753750, -55.5675545, 24.5429535, -80.0289459, 80.1159286
33: -73.7629013, 31.8775139, -73.5814667, 31.8966751, -105.1204681, 104.9048004
34: -63.7367477, 17.9781094, -63.5989113, 17.9339218, -81.3571777, 81.1719894
35: -60.8407211, 24.4055138, -60.7597618, 24.5123100, -84.7876282, 84.5303955
36: -60.8927078, 25.3494759, -60.8045502, 25.3053761, -86.1973114, 86.1524506
37: -89.4384918, 18.7626362, -89.1049347, 18.4492283, -107.7264252, 107.6893463
38: -69.8025208, 29.0799732, -69.7168808, 29.0254517, -98.8279724, 98.7968521
39: -83.4216919, 30.7839851, -83.2324524, 30.7519131, -114.1736069, 114.0164337
40: -65.7975235, 21.6641922, -65.5027924, 21.3007660, -87.0806046, 87.1436615
41: -58.7329674, 28.8240662, -58.4793701, 28.5051079, -87.2380753, 87.3034363
42: -40.2034607, 24.7516747, -40.2015457, 24.5644646, -64.7679291, 64.9532166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1338689, upper bound: 38.1807752
time: 176.57 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1344715, upper bound: 38.2075006
time: 79.63 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -57.3051758, 42.9107971, -57.2125778, 42.8800201, -100.1851959, 100.1233749
1: -26.5947819, 35.2488441, -26.5154667, 35.2307930, -61.8255768, 61.7643127
2: -24.4726906, 37.0070496, -24.3954144, 37.0480576, -61.5207481, 61.4024658
3: -28.7289867, 41.3673630, -28.5913944, 41.3450813, -70.0740662, 69.9587555
4: -31.6893768, 41.2557602, -31.6009560, 41.3222580, -73.0116348, 72.8567200
5: -28.0943680, 42.6622467, -27.9867172, 42.6574593, -70.7518311, 70.6489639
6: -55.2201767, 27.2989044, -55.2426453, 27.2423096, -82.4624863, 82.5415497
7: -32.5346451, 40.4942093, -32.4382477, 40.4436569, -72.9783020, 72.9324570
8: -37.2471008, 49.3601990, -37.1161499, 49.3697586, -86.6168594, 86.4763489
9: -30.3247871, 38.3784103, -30.4000320, 38.3641357, -68.6889191, 68.7784424
10: -49.5996323, 48.1957169, -49.8550148, 48.1079483, -97.7075806, 98.0507355
11: -48.5720711, 29.1017380, -48.7689934, 29.0799961, -77.6520691, 77.8707275
12: -59.8290787, 31.4598389, -60.0935669, 31.3638954, -90.7584229, 91.1157532
13: -51.4632492, 46.9682312, -51.3980103, 46.9004822, -98.3637314, 98.3662415
14: -79.7777786, 42.5811234, -79.8226166, 42.4102592, -122.1880341, 122.4037399
15: -38.1880417, 35.1948242, -38.1599808, 35.2901115, -73.4781494, 73.3548050
16: -48.6462059, 37.2558670, -48.7453880, 37.2335281, -85.8797302, 86.0012512
17: -79.7153702, 34.0733299, -79.7464600, 33.9651604, -113.6805267, 113.8197937
18: -48.2733498, 33.4739418, -48.2731476, 33.3981323, -81.6714783, 81.7470856
19: -38.3014145, 19.2846260, -38.2860565, 19.2525043, -57.5539169, 57.5706825
20: -34.7990723, 24.9674187, -34.8301125, 24.9313984, -59.7304688, 59.7975311
21: -46.2530327, 24.9127464, -46.3317299, 24.8971272, -71.1501617, 71.2444763
22: -49.1757698, 25.2098866, -49.1567726, 25.2208748, -74.3966446, 74.3666611
23: -37.8783264, 26.3690376, -37.8657150, 26.3392773, -64.2176056, 64.2347565
24: -45.4978371, 28.8608475, -45.4594498, 28.8918114, -74.3896484, 74.3202972
25: -39.7463684, 29.4503021, -39.7255783, 29.4446449, -69.1910095, 69.1758804
26: -56.0639572, 38.8191376, -56.1090546, 38.7608490, -94.8248062, 94.9281921
27: -46.0808105, 30.1632156, -46.0629425, 30.2359028, -76.3167114, 76.2261581
28: -37.0156021, 29.8608475, -36.9967041, 29.8717842, -66.8873901, 66.8575516
29: -51.2463913, 24.6087990, -51.2365990, 24.5936852, -75.8400726, 75.8453979
30: -46.3648300, 33.4670868, -46.3569260, 33.4280090, -79.7928391, 79.8240128
31: -49.1524429, 27.7858982, -49.1227570, 27.7349148, -76.8873596, 76.9086533
32: -55.6072235, 24.6753750, -55.6542969, 24.6241245, -80.1050644, 80.2006378
33: -73.7629013, 31.8775139, -73.6769257, 31.9655800, -105.1747742, 104.9835052
34: -63.7367477, 17.9781094, -63.7124748, 18.0477619, -81.4402542, 81.2485657
35: -60.8407211, 24.4055138, -60.8155479, 24.5487022, -84.8210449, 84.5851822
36: -60.8927078, 25.3494759, -60.8853569, 25.3799248, -86.2715378, 86.2329559
37: -89.4384918, 18.7626362, -89.3109665, 18.6221142, -107.8913116, 107.8851547
38: -69.8025208, 29.0799732, -69.7854919, 29.0745640, -98.8770828, 98.8654633
39: -83.4216919, 30.7839851, -83.3356018, 30.7872982, -114.2089920, 114.1195831
40: -65.7975235, 21.6641922, -65.7098083, 21.5099659, -87.2818604, 87.3402176
41: -58.7329674, 28.8240662, -58.6589317, 28.6968861, -87.4298553, 87.4830017
42: -40.2034607, 24.7516747, -40.2931519, 24.6713448, -64.8748016, 65.0448303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=506, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1004078, upper bound: 38.1807754
time: 83.58 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1004078, upper bound: 38.2075008
time: 79.29 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -57.2088013, 42.8804932, -57.1902351, 42.8724670, -100.0812683, 100.0707245
1: -26.5040894, 35.2182693, -26.5054798, 35.2349358, -61.7390251, 61.7237473
2: -24.3812923, 36.9702454, -24.3528214, 37.0334129, -61.4147034, 61.3230667
3: -28.6475983, 41.3627777, -28.6204224, 41.4100456, -70.0576477, 69.9832001
4: -31.5644722, 41.2149963, -31.5588818, 41.3143005, -72.8787689, 72.7738800
5: -28.0208054, 42.6387787, -27.9937096, 42.6815338, -70.7023392, 70.6324921
6: -55.1749153, 27.2127495, -55.2332535, 27.1776123, -82.3525238, 82.4459991
7: -32.4613419, 40.4889679, -32.4596176, 40.4659424, -72.9272842, 72.9485855
8: -37.0576019, 49.2890091, -37.0788727, 49.3982048, -86.4558105, 86.3678818
9: -30.2501450, 38.3327866, -30.3599243, 38.3446732, -68.5948181, 68.6927109
10: -49.5437164, 48.1391258, -49.8658104, 48.1224098, -97.6661224, 98.0049362
11: -48.4384613, 28.9710941, -48.7830124, 29.0043068, -77.4427643, 77.7541046
12: -59.7898636, 31.3667412, -60.0670242, 31.3301334, -90.7009048, 91.0152588
13: -51.3505325, 46.9105377, -51.3209152, 46.9151344, -98.2656708, 98.2314529
14: -79.6890106, 42.6248016, -79.7943115, 42.4171333, -122.1061401, 122.4191132
15: -38.0650749, 35.1392822, -38.1009254, 35.2885170, -73.3535919, 73.2402039
16: -48.5976486, 37.2027550, -48.7610703, 37.2003670, -85.7980194, 85.9638214
17: -79.5806122, 34.0737610, -79.7778168, 33.9733315, -113.5539398, 113.8515778
18: -48.1833992, 33.3186531, -48.2966194, 33.3480492, -81.5314484, 81.6152725
19: -38.2422371, 19.2278290, -38.3128700, 19.2472973, -57.4895325, 57.5406990
20: -34.7547607, 24.9172173, -34.8275490, 24.9106331, -59.6653938, 59.7447662
21: -46.1544418, 24.8259754, -46.3357620, 24.8550472, -71.0094910, 71.1617355
22: -49.0938454, 25.1523705, -49.1294479, 25.2060604, -74.2999039, 74.2818146
23: -37.8218384, 26.2775269, -37.8777237, 26.3009071, -64.1227417, 64.1552505
24: -45.4356384, 28.7663155, -45.4713783, 28.8572197, -74.2928619, 74.2376938
25: -39.7026291, 29.3871613, -39.7198601, 29.4139481, -69.1165771, 69.1070251
26: -55.9649544, 38.7186356, -56.0577583, 38.6943550, -94.6593094, 94.7763977
27: -46.0131264, 30.1016731, -46.0522499, 30.1764240, -76.1895523, 76.1539230
28: -36.9443054, 29.7663364, -36.9888229, 29.8445892, -66.7888947, 66.7551575
29: -51.1444702, 24.5376320, -51.2283783, 24.5608101, -75.7052765, 75.7660065
30: -46.2601852, 33.3253326, -46.3640289, 33.3690338, -79.6292191, 79.6893616
31: -49.0924530, 27.6868896, -49.1534348, 27.7234268, -76.8158798, 76.8403244
32: -55.5728912, 24.5961800, -55.6444016, 24.5916080, -80.0423355, 80.1170654
33: -73.7595139, 31.8242435, -73.7119293, 32.0059776, -105.2495193, 104.9619141
34: -63.6628342, 17.8718185, -63.6446381, 17.9750881, -81.3249664, 81.1352386
35: -60.7813988, 24.3197212, -60.8008957, 24.5454788, -84.7659302, 84.5047684
36: -60.8475380, 25.2918129, -60.8420753, 25.3434486, -86.1902695, 86.1327515
37: -89.4221115, 18.6784515, -89.3340225, 18.6336613, -107.8971329, 107.8416290
38: -69.7562332, 29.0414200, -69.7979889, 29.0903339, -98.8465652, 98.8394089
39: -83.3978195, 30.7182293, -83.3885117, 30.8654022, -114.2632217, 114.1067429
40: -65.7985001, 21.6034050, -65.7003632, 21.4877090, -87.2733765, 87.2813950
41: -58.7185287, 28.7440414, -58.6430168, 28.6296196, -87.3481445, 87.3870544
42: -40.1960640, 24.7142162, -40.2997551, 24.6537857, -64.8498535, 65.0139694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1812675
time: 86.58 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
time: 170.91 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -57.2088013, 42.8804932, -57.3446198, 42.9654579, -100.1742554, 100.2251129
1: -26.5040894, 35.2182693, -26.6067066, 35.3246307, -61.8287201, 61.8249741
2: -24.3812923, 36.9702454, -24.4735699, 37.1420708, -61.5233612, 61.4438171
3: -28.6475983, 41.3627777, -28.7238941, 41.4992447, -70.1468430, 70.0866699
4: -31.5644722, 41.2149963, -31.6747398, 41.4293098, -72.9937820, 72.8897400
5: -28.0208054, 42.6387787, -28.1005001, 42.7868538, -70.8076630, 70.7392807
6: -55.1749153, 27.2127495, -55.3240356, 27.2906933, -82.4656067, 82.5367889
7: -32.4613419, 40.4889679, -32.5829620, 40.5718231, -73.0331650, 73.0719299
8: -37.0576019, 49.2890091, -37.2212029, 49.5376320, -86.5952301, 86.5102081
9: -30.2501450, 38.3327866, -30.4391899, 38.4056969, -68.6558380, 68.7719727
10: -49.5437164, 48.1391258, -49.9619827, 48.2099953, -97.7537079, 98.1011047
11: -48.4384613, 28.9710941, -48.8820648, 29.0704269, -77.5088882, 77.8531570
12: -59.7898636, 31.3667412, -60.1784363, 31.4697189, -90.8288040, 91.1146927
13: -51.3505325, 46.9105377, -51.4967384, 47.0561104, -98.4066467, 98.4072723
14: -79.6890106, 42.6248016, -80.0374756, 42.6484413, -122.3374481, 122.6622772
15: -38.0650749, 35.1392822, -38.1949005, 35.3640518, -73.4291229, 73.3341827
16: -48.5976486, 37.2027550, -48.8619576, 37.2601929, -85.8578415, 86.0647125
17: -79.5806122, 34.0737610, -79.9629822, 34.1548080, -113.7354202, 114.0367432
18: -48.1833992, 33.3186531, -48.3716583, 33.4553642, -81.6387634, 81.6903076
19: -38.2422371, 19.2278290, -38.3740425, 19.2861404, -57.5283775, 57.6018715
20: -34.7547607, 24.9172173, -34.8837280, 24.9701366, -59.7248993, 59.8009453
21: -46.1544418, 24.8259754, -46.4000053, 24.9014454, -71.0558853, 71.2259827
22: -49.0938454, 25.1523705, -49.1959229, 25.2763214, -74.3701630, 74.3482971
23: -37.8218384, 26.2775269, -37.9540901, 26.3560581, -64.1778946, 64.2316132
24: -45.4356384, 28.7663155, -45.5374870, 28.8996925, -74.3353271, 74.3038025
25: -39.7026291, 29.3871613, -39.7789307, 29.4771652, -69.1797943, 69.1660919
26: -55.9649544, 38.7186356, -56.1528015, 38.8142548, -94.7792053, 94.8714371
27: -46.0131264, 30.1016731, -46.1219254, 30.2576790, -76.2708054, 76.2236023
28: -36.9443054, 29.7663364, -37.0455933, 29.8993263, -66.8436279, 66.8119278
29: -51.1444702, 24.5376320, -51.2899780, 24.6193314, -75.7638016, 75.8276062
30: -46.2601852, 33.3253326, -46.4211349, 33.4476089, -79.7077942, 79.7464676
31: -49.0924530, 27.6868896, -49.2419281, 27.7709599, -76.8634109, 76.9288177
32: -55.5728912, 24.5961800, -55.7312546, 24.6730194, -80.1186676, 80.2018433
33: -73.7595139, 31.8242435, -73.8074951, 32.0748329, -105.3038712, 105.0407410
34: -63.6628342, 17.8718185, -63.7582436, 18.0889111, -81.4080887, 81.2118988
35: -60.7813988, 24.3197212, -60.8568230, 24.5819130, -84.7994080, 84.5596771
36: -60.8475380, 25.2918129, -60.9229813, 25.4179955, -86.2644958, 86.2133255
37: -89.4221115, 18.6784515, -89.5400696, 18.8065166, -108.0619507, 108.0374222
38: -69.7562332, 29.0414200, -69.8666077, 29.1395416, -98.8957748, 98.9080276
39: -83.3978195, 30.7182293, -83.4920425, 30.9008141, -114.2986298, 114.2102737
40: -65.7985001, 21.6034050, -65.9075089, 21.6968613, -87.4745941, 87.4780655
41: -58.7185287, 28.7440414, -58.8225441, 28.8213310, -87.5398560, 87.5665894
42: -40.1960640, 24.7142162, -40.3911781, 24.7606754, -64.9567413, 65.1053925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1181418
time: 856.36 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
time: 103.33 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -57.3196907, 42.9396133, -57.2253571, 42.8864365, -100.2061310, 100.1649704
1: -26.5993462, 35.2774277, -26.5347347, 35.2405930, -61.8399391, 61.8121643
2: -24.4783306, 37.0360107, -24.3858547, 37.0390015, -61.5173340, 61.4218674
3: -28.7346649, 41.4155121, -28.6494274, 41.4164124, -70.1510773, 70.0649414
4: -31.6953049, 41.2867889, -31.6034145, 41.3200111, -73.0153198, 72.8902054
5: -28.1003609, 42.7021179, -28.0196857, 42.6874275, -70.7877884, 70.7218018
6: -55.2414398, 27.3072243, -55.2430725, 27.2097912, -82.4512329, 82.5502930
7: -32.5423164, 40.5347595, -32.4831009, 40.4709396, -73.0132599, 73.0178604
8: -37.2538948, 49.4113426, -37.1476707, 49.4090042, -86.6629028, 86.5590134
9: -30.3334942, 38.3857040, -30.3877983, 38.3513947, -68.6848907, 68.7734985
10: -49.6273727, 48.2043343, -49.8906403, 48.1351967, -97.7625732, 98.0949707
11: -48.5899124, 29.1065941, -48.7976379, 29.0512581, -77.6411743, 77.9042358
12: -59.8548584, 31.4747467, -60.0750275, 31.3654404, -90.8014374, 91.1204681
13: -51.4736176, 47.0167542, -51.3613930, 46.9326019, -98.4062195, 98.3781433
14: -79.7899628, 42.6606903, -79.8276215, 42.4323807, -122.2223434, 122.4883118
15: -38.1978531, 35.2080574, -38.1453056, 35.2963142, -73.4941711, 73.3533630
16: -48.6687431, 37.2615662, -48.7753334, 37.2195663, -85.8883057, 86.0368958
17: -79.7243652, 34.1334152, -79.7977600, 33.9926529, -113.7170181, 113.9311752
18: -48.2996826, 33.4808655, -48.3110237, 33.4051743, -81.7048569, 81.7918854
19: -38.3240509, 19.2873592, -38.3237152, 19.2689476, -57.5929985, 57.6110764
20: -34.8062897, 24.9838009, -34.8391609, 24.9333878, -59.7396774, 59.8229599
21: -46.2656975, 24.9169559, -46.3497200, 24.8874950, -71.1531906, 71.2666779
22: -49.1842957, 25.2192383, -49.1440201, 25.2282600, -74.4125519, 74.3632584
23: -37.8986053, 26.3740768, -37.8872147, 26.3335247, -64.2321320, 64.2612915
24: -45.5186539, 28.8642426, -45.4834976, 28.8908730, -74.4095306, 74.3477402
25: -39.7579002, 29.4574966, -39.7305222, 29.4365826, -69.1944809, 69.1880188
26: -56.0749359, 38.8286552, -56.0730362, 38.7340546, -94.8089905, 94.9016876
27: -46.0947762, 30.1688824, -46.0675697, 30.1993637, -76.2941437, 76.2364502
28: -37.0246658, 29.8739738, -36.9984207, 29.8815556, -66.9062195, 66.8723907
29: -51.2543144, 24.6146393, -51.2432289, 24.5872498, -75.8415680, 75.8578644
30: -46.3750648, 33.4793167, -46.3775940, 33.4217529, -79.7968140, 79.8569107
31: -49.1834869, 27.7896729, -49.1671944, 27.7591763, -76.9426651, 76.9568634
32: -55.6260948, 24.6847687, -55.6530609, 24.6190739, -80.1227264, 80.2089386
33: -73.8032227, 31.8845711, -73.7234344, 32.0216827, -105.2972183, 105.0468063
34: -63.7483139, 17.9840508, -63.6550713, 18.0088863, -81.4441757, 81.2258606
35: -60.8509331, 24.4096355, -60.8107452, 24.5719948, -84.8623581, 84.5814209
36: -60.8992195, 25.3551331, -60.8487015, 25.3629951, -86.2615356, 86.2023621
37: -89.5073090, 18.7688217, -89.3456116, 18.6628418, -108.0117950, 107.9336166
38: -69.8164368, 29.0895863, -69.8095474, 29.1007214, -98.9171600, 98.8991318
39: -83.4663467, 30.7896748, -83.4073486, 30.8721008, -114.3384476, 114.1970215
40: -65.8561783, 21.6710930, -65.7136230, 21.5059242, -87.3472748, 87.3595428
41: -58.7827263, 28.8301468, -58.6530457, 28.6585598, -87.4412842, 87.4831924
42: -40.2302704, 24.7598667, -40.3086929, 24.6674271, -64.8976974, 65.0685577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1812675
time: 89.83 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
time: 101.91 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -57.3196907, 42.9396133, -57.3797531, 42.9794388, -100.2991333, 100.3193665
1: -26.5993462, 35.2774277, -26.6359501, 35.3302689, -61.9296150, 61.9133759
2: -24.4783306, 37.0360107, -24.5066032, 37.1476707, -61.6259995, 61.5426140
3: -28.7346649, 41.4155121, -28.7529087, 41.5055771, -70.2402420, 70.1684189
4: -31.6953049, 41.2867889, -31.7192631, 41.4350204, -73.1303253, 73.0060501
5: -28.1003609, 42.7021179, -28.1264668, 42.7927094, -70.8930664, 70.8285828
6: -55.2414398, 27.3072243, -55.3338623, 27.3228912, -82.5643311, 82.6410828
7: -32.5423164, 40.5347595, -32.6064606, 40.5768280, -73.1191406, 73.1412201
8: -37.2538948, 49.4113426, -37.2900238, 49.5484390, -86.8023376, 86.7013702
9: -30.3334942, 38.3857040, -30.4670563, 38.4124146, -68.7459106, 68.8527603
10: -49.6273727, 48.2043343, -49.9867935, 48.2227974, -97.8501740, 98.1911316
11: -48.5899124, 29.1065941, -48.8966827, 29.1173897, -77.7073059, 78.0032806
12: -59.8548584, 31.4747467, -60.1864090, 31.5050182, -90.9293747, 91.2198639
13: -51.4736176, 47.0167542, -51.5372467, 47.0735741, -98.5471954, 98.5540009
14: -79.7899628, 42.6606903, -80.0707703, 42.6636429, -122.4536057, 122.7314606
15: -38.1978531, 35.2080574, -38.2392769, 35.3718796, -73.5697327, 73.4473343
16: -48.6687431, 37.2615662, -48.8761978, 37.2794266, -85.9481659, 86.1377640
17: -79.7243652, 34.1334152, -79.9829712, 34.1741562, -113.8985214, 114.1163864
18: -48.2996826, 33.4808655, -48.3860512, 33.5125122, -81.8121948, 81.8669128
19: -38.3240509, 19.2873592, -38.3848686, 19.3077965, -57.6318474, 57.6722260
20: -34.8062897, 24.9838009, -34.8953362, 24.9928741, -59.7991638, 59.8791351
21: -46.2656975, 24.9169559, -46.4139633, 24.9339142, -71.1996155, 71.3309174
22: -49.1842957, 25.2192383, -49.2104950, 25.2985249, -74.4828186, 74.4297333
23: -37.8986053, 26.3740768, -37.9635696, 26.3886852, -64.2872925, 64.3376465
24: -45.5186539, 28.8642426, -45.5495872, 28.9333420, -74.4519958, 74.4138336
25: -39.7579002, 29.4574966, -39.7895889, 29.4997749, -69.2576752, 69.2470856
26: -56.0749359, 38.8286552, -56.1680641, 38.8538818, -94.9288177, 94.9967194
27: -46.0947762, 30.1688824, -46.1372337, 30.2806225, -76.3753967, 76.3061142
28: -37.0246658, 29.8739738, -37.0551910, 29.9362526, -66.9609222, 66.9291687
29: -51.2543144, 24.6146393, -51.3048477, 24.6457844, -75.9001007, 75.9194870
30: -46.3750648, 33.4793167, -46.4347076, 33.5003662, -79.8754272, 79.9140244
31: -49.1834869, 27.7896729, -49.2556953, 27.8067284, -76.9902191, 77.0453644
32: -55.6260948, 24.6847687, -55.7399368, 24.7005005, -80.1990738, 80.2937393
33: -73.8032227, 31.8845711, -73.8190384, 32.0906029, -105.3515930, 105.1256332
34: -63.7483139, 17.9840508, -63.7686691, 18.1227283, -81.5273209, 81.3025055
35: -60.8509331, 24.4096355, -60.8666687, 24.6084099, -84.8958359, 84.6363525
36: -60.8992195, 25.3551331, -60.9295883, 25.4375610, -86.3357391, 86.2828979
37: -89.5073090, 18.7688217, -89.5515976, 18.8357468, -108.1766357, 108.1294022
38: -69.8164368, 29.0895863, -69.8781891, 29.1499367, -98.9663696, 98.9677734
39: -83.4663467, 30.7896748, -83.5109024, 30.9075317, -114.3738785, 114.3005753
40: -65.8561783, 21.6710930, -65.9207077, 21.7151146, -87.5485001, 87.5562134
41: -58.7827263, 28.8301468, -58.8325691, 28.8502884, -87.6330109, 87.6627197
42: -40.2302704, 24.7598667, -40.4001236, 24.7743149, -65.0045853, 65.1599884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=246, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=507, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 725
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 940
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 651
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 658
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 562
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 680

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1719

## Relational analysis of IS_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1812676
time: 79.81 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
time: 97.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 180.09 seconds
IS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2072686
IS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2072686
IS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2072688
IS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.0730385, upper bound: 38.2075009
IS_A2_B2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1338689, upper bound: 38.1807752
IS_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1344715, upper bound: 38.2075006
IS_A2_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1004078, upper bound: 38.1807754
IS_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1004078, upper bound: 38.2075008
IS_A2_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1812675
IS_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
IS_A2_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1181418
IS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
IS_A2_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1812675
IS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
IS_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.1812676
IS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 180.09
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524

## BFS IS instance: IS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -57.1229439, 42.8078613, -56.9984360, 42.7582703, -99.8812103, 99.8062973
1: -26.4376259, 35.1760941, -26.3635521, 35.1308098, -61.5684357, 61.5396461
2: -24.3115959, 36.9287949, -24.2201443, 36.9296227, -61.2412186, 61.1489410
3: -28.5773525, 41.3001900, -28.4369030, 41.2446594, -69.8220139, 69.7370911
4: -31.4581394, 41.1707153, -31.4068623, 41.1968231, -72.6549606, 72.5775757
5: -27.9626293, 42.5854111, -27.8362770, 42.5417023, -70.5043335, 70.4216919
6: -55.1316185, 27.1428299, -55.1345863, 27.0762901, -82.2079086, 82.2774200
7: -32.4025230, 40.4372635, -32.2736893, 40.3288956, -72.7314148, 72.7109528
8: -36.8907776, 49.2127457, -36.8511353, 49.2110176, -86.1017914, 86.0638809
9: -30.1915855, 38.3083420, -30.2763176, 38.2903671, -68.4819489, 68.5846558
10: -49.4727249, 48.1008797, -49.7194176, 47.9975014, -97.4702301, 97.8202972
11: -48.3863945, 28.8596497, -48.6433029, 28.9309196, -77.3173141, 77.5029526
12: -59.7465591, 31.2657433, -59.9691772, 31.1598358, -90.4831696, 90.8247757
13: -51.2864380, 46.8257942, -51.1633224, 46.7295227, -98.0159607, 97.9891205
14: -79.6164856, 42.4974442, -79.5253143, 42.1475029, -121.7639923, 122.0227585
15: -37.9537544, 35.1093979, -37.9874115, 35.2008972, -73.1546478, 73.0968094
16: -48.5440331, 37.1339874, -48.6196213, 37.1329346, -85.6769714, 85.7536087
17: -79.5249176, 33.9558144, -79.5250244, 33.7445145, -113.2694321, 113.4808350
18: -48.1245804, 33.1985741, -48.1726151, 33.1955338, -81.3201141, 81.3711853
19: -38.1957169, 19.1862259, -38.2059860, 19.1788425, -57.3745575, 57.3922119
20: -34.7170181, 24.8570251, -34.7517815, 24.8341484, -59.5511665, 59.6088066
21: -46.1109085, 24.7640209, -46.2429085, 24.7983456, -70.9092560, 71.0069275
22: -49.0517960, 25.1095638, -49.0643463, 25.1170750, -74.1688690, 74.1739120
23: -37.7797012, 26.2075691, -37.7721329, 26.2297382, -64.0094376, 63.9797020
24: -45.3877411, 28.6963730, -45.3720093, 28.7929268, -74.1806641, 74.0683823
25: -39.6656113, 29.3397751, -39.6470261, 29.3451996, -69.0108109, 68.9868011
26: -55.9136276, 38.6498032, -55.9848709, 38.5805626, -94.4941864, 94.6346741
27: -45.9650803, 30.0738354, -45.9663200, 30.1241131, -76.0891953, 76.0401535
28: -36.9135742, 29.6862335, -36.9227791, 29.7571011, -66.6706772, 66.6090088
29: -51.1057777, 24.4842892, -51.1495590, 24.4925613, -75.5983429, 75.6338501
30: -46.2186356, 33.2044067, -46.2756119, 33.2596512, -79.4782867, 79.4800186
31: -49.0295830, 27.6112709, -49.0095863, 27.6273098, -76.6568909, 76.6208572
32: -55.5355301, 24.5307026, -55.5525513, 24.4960785, -79.9106369, 79.9630585
33: -73.6930466, 31.7741337, -73.5610046, 31.8661728, -105.0289230, 104.7729187
34: -63.6267281, 17.7964706, -63.5801315, 17.8761597, -81.1903839, 80.9952240
35: -60.7484894, 24.2442493, -60.7421303, 24.4617119, -84.6441345, 84.3746643
36: -60.8246918, 25.2435226, -60.7923508, 25.2712364, -86.0950470, 86.0345383
37: -89.3268509, 18.6027355, -89.0842972, 18.3963966, -107.5611038, 107.5196838
38: -69.7139587, 29.0030384, -69.6955719, 29.0051289, -98.7190857, 98.6986084
39: -83.3187790, 30.6946259, -83.2018433, 30.7389412, -114.0577240, 113.8964691
40: -65.7109222, 21.5484676, -65.4795532, 21.2660122, -86.9616547, 87.0090790
41: -58.6493263, 28.6931763, -58.4627075, 28.4606514, -87.1099777, 87.1558838
42: -40.1484070, 24.6799889, -40.1854553, 24.5418491, -64.6902542, 64.8654480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=246, inp2_unstable=246, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=503, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 654
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 577
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2072688
time: 98.31 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1470857
time: 94.51 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -57.1958580, 42.8546066, -57.0184517, 42.7712402, -99.9671021, 99.8730621
1: -26.5094948, 35.2313271, -26.3833866, 35.1347504, -61.6442451, 61.6147156
2: -24.3780060, 36.9870148, -24.2399788, 36.9333191, -61.3113251, 61.2269936
3: -28.6460342, 41.3467941, -28.4572124, 41.2490120, -69.8950500, 69.8040085
4: -31.5625496, 41.2463608, -31.4384270, 41.2005997, -72.7631531, 72.6847839
5: -28.0185966, 42.6403885, -27.8510666, 42.5457001, -70.5643005, 70.4914551
6: -55.1872787, 27.2097206, -55.1410141, 27.0953217, -82.2826004, 82.3507385
7: -32.4681473, 40.4752655, -32.2896118, 40.3322258, -72.8003693, 72.7648773
8: -37.0506287, 49.3428307, -36.9016800, 49.2185783, -86.2692108, 86.2445068
9: -30.2431297, 38.3516693, -30.2917995, 38.2955246, -68.5386505, 68.6434708
10: -49.5266037, 48.1537361, -49.7326660, 48.0064697, -97.5330734, 97.8863983
11: -48.5298004, 28.9671936, -48.6531944, 28.9647789, -77.4945831, 77.6203918
12: -59.8052597, 31.3642197, -59.9744568, 31.1869011, -90.5694046, 90.9201050
13: -51.3509979, 46.8843994, -51.1797714, 46.7406960, -98.0916901, 98.0641708
14: -79.6744766, 42.5480995, -79.5399780, 42.1620636, -121.8365402, 122.0880737
15: -38.0641251, 35.1789246, -38.0194702, 35.2057877, -73.2699127, 73.1983948
16: -48.6049690, 37.1988602, -48.6286736, 37.1521683, -85.7571411, 85.8275299
17: -79.6593094, 34.0156631, -79.5391541, 33.7620277, -113.4213409, 113.5548172
18: -48.2198753, 33.3113937, -48.1826286, 33.2311478, -81.4510193, 81.4940186
19: -38.2644730, 19.2237720, -38.2131958, 19.1900215, -57.4544945, 57.4369659
20: -34.7546234, 24.8957844, -34.7609711, 24.8450432, -59.5996666, 59.6567535
21: -46.2067947, 24.8244267, -46.2521820, 24.8169441, -71.0237427, 71.0766068
22: -49.1195183, 25.1451187, -49.0742111, 25.1271782, -74.2466965, 74.2193298
23: -37.8428879, 26.2739487, -37.7788467, 26.2487679, -64.0916595, 64.0527954
24: -45.4581871, 28.7629051, -45.3800774, 28.8139820, -74.2721710, 74.1429825
25: -39.7131958, 29.3818932, -39.6548805, 29.3567848, -69.0699768, 69.0367737
26: -55.9897614, 38.7070351, -55.9971962, 38.5984573, -94.5882187, 94.7042313
27: -46.0152588, 30.0977249, -45.9763985, 30.1305046, -76.1457672, 76.0741272
28: -36.9759369, 29.7519989, -36.9292603, 29.7769909, -66.7529297, 66.6812592
29: -51.1933441, 24.5321312, -51.1583786, 24.5072441, -75.7005920, 75.6905060
30: -46.3210106, 33.3154106, -46.2848358, 33.2943497, -79.6153564, 79.6002502
31: -49.1089554, 27.6827354, -49.0194893, 27.6495323, -76.7584839, 76.7022247
32: -55.5797958, 24.5990849, -55.5575676, 24.5133591, -79.9717102, 80.0330963
33: -73.7407227, 31.8239098, -73.5689926, 31.8772373, -105.0619659, 104.8328476
34: -63.7013893, 17.8778782, -63.5875473, 17.8978806, -81.2837601, 81.0829620
35: -60.8233757, 24.3271027, -60.7491531, 24.4839134, -84.7412720, 84.4576797
36: -60.8777580, 25.2917881, -60.7971497, 25.2845764, -86.1615067, 86.0875854
37: -89.4043503, 18.6784363, -89.0919266, 18.4176197, -107.6599426, 107.5983810
38: -69.7675552, 29.0464916, -69.7041702, 29.0135937, -98.7811508, 98.7506638
39: -83.3713531, 30.7503242, -83.2121887, 30.7442207, -114.1155701, 113.9625092
40: -65.7518158, 21.6080551, -65.4876251, 21.2796726, -87.0131454, 87.0752411
41: -58.6922455, 28.7439899, -58.4684105, 28.4744778, -87.1667252, 87.2124023
42: -40.1773376, 24.7170715, -40.1915016, 24.5498428, -64.7271805, 64.9085693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=246, inp2_unstable=246, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2072684
time: 96.03 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1473994
time: 164.48 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -57.1229439, 42.8078613, -57.1527405, 42.8512535, -99.9741974, 99.9606018
1: -26.4376259, 35.1760941, -26.4647808, 35.2204208, -61.6580467, 61.6408768
2: -24.3115959, 36.9287949, -24.3409061, 37.0381889, -61.3497849, 61.2696991
3: -28.5773525, 41.3001900, -28.5404320, 41.3337364, -69.9110870, 69.8406219
4: -31.4581394, 41.1707153, -31.5227337, 41.3118858, -72.7700272, 72.6934509
5: -27.9626293, 42.5854111, -27.9430943, 42.6469345, -70.6095657, 70.5285034
6: -55.1316185, 27.1428299, -55.2252769, 27.1891537, -82.3207703, 82.3681030
7: -32.4025230, 40.4372635, -32.3970947, 40.4348145, -72.8373413, 72.8343582
8: -36.8907776, 49.2127457, -36.9934998, 49.3502922, -86.2410736, 86.2062454
9: -30.1915855, 38.3083420, -30.3553162, 38.3514519, -68.5430374, 68.6636581
10: -49.4727249, 48.1008797, -49.8155174, 48.0849762, -97.5577011, 97.9163971
11: -48.3863945, 28.8596497, -48.7423172, 28.9970436, -77.3834381, 77.6019669
12: -59.7465591, 31.2657433, -60.0795670, 31.2994385, -90.6109619, 90.9234238
13: -51.2864380, 46.8257942, -51.3391457, 46.8705750, -98.1570129, 98.1649399
14: -79.6164856, 42.4974442, -79.7684555, 42.3787422, -121.9952240, 122.2658997
15: -37.9537544, 35.1093979, -38.0813446, 35.2763672, -73.2301178, 73.1907425
16: -48.5440331, 37.1339874, -48.7203903, 37.1927338, -85.7367706, 85.8543777
17: -79.5249176, 33.9558144, -79.7102814, 33.9260979, -113.4510193, 113.6660919
18: -48.1245804, 33.1985741, -48.2475967, 33.3028908, -81.4274750, 81.4461670
19: -38.1957169, 19.1862259, -38.2668495, 19.2176666, -57.4133835, 57.4530754
20: -34.7170181, 24.8570251, -34.8079453, 24.8936272, -59.6106453, 59.6649704
21: -46.1109085, 24.7640209, -46.3069534, 24.8447323, -70.9556427, 71.0709763
22: -49.0517960, 25.1095638, -49.1307297, 25.1873302, -74.2391281, 74.2402954
23: -37.7797012, 26.2075691, -37.8485184, 26.2848701, -64.0645752, 64.0560913
24: -45.3877411, 28.6963730, -45.4379158, 28.8353424, -74.2230835, 74.1342926
25: -39.6656113, 29.3397751, -39.7060394, 29.4083900, -69.0740051, 69.0458145
26: -55.9136276, 38.6498032, -56.0799255, 38.7005539, -94.6141815, 94.7297287
27: -45.9650803, 30.0738354, -46.0359879, 30.2052917, -76.1703720, 76.1098251
28: -36.9135742, 29.6862335, -36.9795609, 29.8118763, -66.7254486, 66.6657944
29: -51.1057777, 24.4842892, -51.2111549, 24.5511112, -75.6568909, 75.6954422
30: -46.2186356, 33.2044067, -46.3324623, 33.3381004, -79.5567322, 79.5368652
31: -49.0295830, 27.6112709, -49.0979385, 27.6747742, -76.7043610, 76.7092133
32: -55.5355301, 24.5307026, -55.6392403, 24.5772228, -79.9866943, 80.0477371
33: -73.6930466, 31.7741337, -73.6564484, 31.9349976, -105.0832367, 104.8516083
34: -63.6267281, 17.7964706, -63.6936722, 17.9900265, -81.2734375, 81.0717926
35: -60.7484894, 24.2442493, -60.7979431, 24.4980640, -84.6775970, 84.4294739
36: -60.8246918, 25.2435226, -60.8731613, 25.3457718, -86.1692352, 86.1150284
37: -89.3268509, 18.6027355, -89.2902908, 18.5693111, -107.7259598, 107.7154312
38: -69.7139587, 29.0030384, -69.7641373, 29.0542259, -98.7681885, 98.7671738
39: -83.3187790, 30.6946259, -83.3049545, 30.7743225, -114.0931015, 113.9995804
40: -65.7109222, 21.5484676, -65.6865997, 21.4751740, -87.1628799, 87.2056732
41: -58.6493263, 28.6931763, -58.6422272, 28.6524200, -87.3017426, 87.3354034
42: -40.1484070, 24.6799889, -40.2770500, 24.6487274, -64.7971344, 64.9570389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=246, inp2_unstable=246, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=503, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 763
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 564
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 530
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2072690
time: 77.10 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1470858
time: 82.02 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -57.1958580, 42.8546066, -57.1727715, 42.8641968, -100.0600586, 100.0273743
1: -26.5094948, 35.2313271, -26.4846287, 35.2243576, -61.7338524, 61.7159576
2: -24.3780060, 36.9870148, -24.3607273, 37.0419121, -61.4199181, 61.3477402
3: -28.6460342, 41.3467941, -28.5607224, 41.3381042, -69.9841385, 69.9075165
4: -31.5625496, 41.2463608, -31.5542755, 41.3156853, -72.8782349, 72.8006363
5: -28.0185966, 42.6403885, -27.9579163, 42.6509323, -70.6695251, 70.5983047
6: -55.1872787, 27.2097206, -55.2317123, 27.2082005, -82.3954773, 82.4414368
7: -32.4681473, 40.4752655, -32.4130249, 40.4381332, -72.9062805, 72.8882904
8: -37.0506287, 49.3428307, -37.0440292, 49.3578644, -86.4084930, 86.3868561
9: -30.2431297, 38.3516693, -30.3708076, 38.3565941, -68.5997238, 68.7224731
10: -49.5266037, 48.1537361, -49.8287354, 48.0939674, -97.6205750, 97.9824677
11: -48.5298004, 28.9671936, -48.7521973, 29.0308914, -77.5606918, 77.7193909
12: -59.8052597, 31.3642197, -60.0848694, 31.3265095, -90.6972122, 91.0187531
13: -51.3509979, 46.8843994, -51.3556442, 46.8817482, -98.2327423, 98.2400436
14: -79.6744766, 42.5480995, -79.7831039, 42.3932648, -122.0677414, 122.3312073
15: -38.0641251, 35.1789246, -38.1134071, 35.2812614, -73.3453827, 73.2923279
16: -48.6049690, 37.1988602, -48.7294121, 37.2120132, -85.8169861, 85.9282684
17: -79.6593094, 34.0156631, -79.7243881, 33.9436493, -113.6029587, 113.7400513
18: -48.2198753, 33.3113937, -48.2575798, 33.3384972, -81.5583725, 81.5689697
19: -38.2644730, 19.2237720, -38.2740707, 19.2288322, -57.4933052, 57.4978409
20: -34.7546234, 24.8957844, -34.8171387, 24.9045258, -59.6591492, 59.7129211
21: -46.2067947, 24.8244267, -46.3161964, 24.8633327, -71.0701294, 71.1406250
22: -49.1195183, 25.1451187, -49.1406059, 25.1974392, -74.3169556, 74.2857208
23: -37.8428879, 26.2739487, -37.8552284, 26.3039131, -64.1468048, 64.1291809
24: -45.4581871, 28.7629051, -45.4459877, 28.8563690, -74.3145599, 74.2088928
25: -39.7131958, 29.3818932, -39.7138824, 29.4199886, -69.1331863, 69.0957794
26: -55.9897614, 38.7070351, -56.0922241, 38.7184410, -94.7082062, 94.7992554
27: -46.0152588, 30.0977249, -46.0460968, 30.2116909, -76.2269516, 76.1438217
28: -36.9759369, 29.7519989, -36.9860344, 29.8317680, -66.8077087, 66.7380371
29: -51.1933441, 24.5321312, -51.2199554, 24.5658150, -75.7591553, 75.7520905
30: -46.3210106, 33.3154106, -46.3416862, 33.3728333, -79.6938477, 79.6570969
31: -49.1089554, 27.6827354, -49.1078491, 27.6969872, -76.8059387, 76.7905884
32: -55.5797958, 24.5990849, -55.6442719, 24.5945396, -80.0477905, 80.1177597
33: -73.7407227, 31.8239098, -73.6644440, 31.9460773, -105.1162720, 104.9115448
34: -63.7013893, 17.8778782, -63.7010880, 18.0117588, -81.3668137, 81.1595459
35: -60.8233757, 24.3271027, -60.8049240, 24.5203152, -84.7747345, 84.5125198
36: -60.8777580, 25.2917881, -60.8779602, 25.3590813, -86.2356796, 86.1680984
37: -89.4043503, 18.6784363, -89.2979736, 18.5905666, -107.8248215, 107.7941895
38: -69.7675552, 29.0464916, -69.7727661, 29.0627079, -98.8302612, 98.8192596
39: -83.3713531, 30.7503242, -83.3153534, 30.7796154, -114.1509705, 114.0656738
40: -65.7518158, 21.6080551, -65.6946869, 21.4888363, -87.2143707, 87.2718048
41: -58.6922455, 28.7439899, -58.6479759, 28.6662216, -87.3584671, 87.3919678
42: -40.1773376, 24.7170715, -40.2830811, 24.6567116, -64.8340454, 65.0001526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=246, inp2_unstable=246, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 733
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 735
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 762
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 725
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 949
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 563
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 626
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 746
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 643
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 966
type: A, layer: 1, pos: 532
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 532
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 561
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 753
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 655
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 737
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 635
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: B, layer: 1, pos: 609
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 610
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2075010
time: 93.11 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1473995
time: 117.88 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -57.3005295, 42.9090042, -57.0596771, 42.7899323, -100.0904617, 99.9686813
1: -26.5932331, 35.2480850, -26.4239826, 35.1827774, -61.7760086, 61.6720657
2: -24.4710541, 37.0064774, -24.2769794, 36.9851341, -61.4561882, 61.2834549
3: -28.7273846, 41.3667068, -28.4919109, 41.2881851, -70.0155716, 69.8586197
4: -31.6872444, 41.2548790, -31.4889545, 41.2695580, -72.9568024, 72.7438354
5: -28.0915241, 42.6616135, -27.8836327, 42.5937271, -70.6852493, 70.5452423
6: -55.2190475, 27.2969837, -55.1857185, 27.1345406, -82.3535919, 82.4827042
7: -32.5330162, 40.4936905, -32.3289566, 40.3645744, -72.8975906, 72.8226471
8: -37.2438278, 49.3590927, -36.9735146, 49.3355446, -86.5793762, 86.3326111
9: -30.3234863, 38.3775711, -30.3225212, 38.3292427, -68.6527252, 68.7000885
10: -49.5981979, 48.1945610, -49.7693748, 48.0436859, -97.6418839, 97.9639359
11: -48.5699883, 29.0996056, -48.7790031, 29.0147820, -77.5847702, 77.8786087
12: -59.8283348, 31.4577694, -60.0243607, 31.2365608, -90.6382751, 91.0568848
13: -51.4614258, 46.9669838, -51.2328415, 46.7818336, -98.2432556, 98.1998291
14: -79.7715988, 42.5793762, -79.5770111, 42.1823425, -121.9539413, 122.1563873
15: -38.1858635, 35.1938248, -38.0747299, 35.2674484, -73.4533081, 73.2685547
16: -48.6445999, 37.2535706, -48.6741943, 37.1754227, -85.8200226, 85.9277649
17: -79.7133942, 34.0711250, -79.6486206, 33.7854233, -113.4988174, 113.7197418
18: -48.2722244, 33.4714737, -48.2609940, 33.2903214, -81.5625458, 81.7324677
19: -38.3002777, 19.2826080, -38.2700272, 19.2123165, -57.5125961, 57.5526352
20: -34.7977371, 24.9632988, -34.7810402, 24.8667927, -59.6645279, 59.7443390
21: -46.2515488, 24.9114265, -46.3325653, 24.8533325, -71.1048813, 71.2439880
22: -49.1741982, 25.2086620, -49.1245728, 25.1526585, -74.3268585, 74.3332367
23: -37.8773193, 26.3663101, -37.8305855, 26.2854691, -64.1627884, 64.1968994
24: -45.4964905, 28.8591003, -45.4368706, 28.8491955, -74.3456879, 74.2959747
25: -39.7453461, 29.4482861, -39.6886864, 29.3831749, -69.1285248, 69.1369705
26: -56.0624275, 38.8163757, -56.0497894, 38.6387711, -94.7012024, 94.8661652
27: -46.0792694, 30.1619415, -46.0094109, 30.1563644, -76.2356339, 76.1713562
28: -37.0145760, 29.8578110, -36.9805756, 29.8157349, -66.8303070, 66.8383865
29: -51.2447052, 24.6073685, -51.2315025, 24.5353718, -75.7800751, 75.8388672
30: -46.3632698, 33.4647408, -46.3709183, 33.3517151, -79.7149811, 79.8356628
31: -49.1512947, 27.7837715, -49.0819511, 27.6868992, -76.8381958, 76.8657227
32: -55.6059151, 24.6732903, -55.5933380, 24.5552006, -80.0377731, 80.1392975
33: -73.7619019, 31.8738880, -73.6030884, 31.9031029, -105.1330109, 104.8943787
34: -63.7358055, 17.9759121, -63.6490440, 17.9456654, -81.3571625, 81.2190933
35: -60.8399849, 24.4036503, -60.8119240, 24.5236187, -84.7903900, 84.5815125
36: -60.8919525, 25.3482838, -60.8412781, 25.3108902, -86.2018890, 86.1880188
37: -89.4370422, 18.7602692, -89.1559906, 18.4552135, -107.7268524, 107.7379456
38: -69.8013763, 29.0785809, -69.7422028, 29.0398655, -98.8412399, 98.8207855
39: -83.4203339, 30.7830582, -83.2504807, 30.7896461, -114.2099762, 114.0335388
40: -65.7957153, 21.6613216, -65.5145264, 21.3121986, -87.0899506, 87.1501999
41: -58.7320518, 28.8224068, -58.5028915, 28.5110817, -87.2431335, 87.3253021
42: -40.2023239, 24.7507095, -40.2098045, 24.5753403, -64.7776642, 64.9605103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=247, inp2_unstable=245, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=504, inp2_unstable=505, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 733
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 734
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 724
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 654
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 735
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 670
type: A, layer: 1, pos: 670
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 762
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 722
type: A, layer: 1, pos: 722
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 763
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 565
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 565
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 622
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 967
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 967
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 564
type: A, layer: 1, pos: 949
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 732
type: B, layer: 1, pos: 732
type: A, layer: 1, pos: 563
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 706
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 706
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 611
type: A, layer: 1, pos: 611
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 626
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 971
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 966
type: B, layer: 1, pos: 643
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 642
type: B, layer: 1, pos: 642
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 593
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 755
type: A, layer: 1, pos: 755
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 689
type: B, layer: 1, pos: 689
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 641
type: A, layer: 1, pos: 641
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 556
type: B, layer: 1, pos: 753
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 556
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 577
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 530
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 615
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 671
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 615
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 655
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 594
type: B, layer: 1, pos: 594
type: A, layer: 1, pos: 737
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 537
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 537
type: B, layer: 1, pos: 640
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 640
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 553
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 672
type: A, layer: 1, pos: 672
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 595
type: B, layer: 1, pos: 595
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 576
type: A, layer: 1, pos: 576
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 609
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 592
type: A, layer: 1, pos: 592
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 546
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 610
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 538
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 624
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 624
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 608
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 842
type: B, layer: 1, pos: 842
type: A, layer: 1, pos: 562
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0363591, upper bound: 38.2075004
time: 129.72 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -38.0363591, upper bound: 38.1205418
time: 100.21 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 232.59 seconds
IS_A2_B2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2072688
IS_A2_B2_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1470857
IS_A2_B2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2072684
IS_A2_B2_B1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1473994
IS_A2_B2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2072690
IS_A2_B2_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1470858
IS_A2_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.2075010
IS_A2_B2_B1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0088975, upper bound: 38.1473995
IS_A2_B2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0363591, upper bound: 38.2075004
IS_A2_B2_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 232.59
Output dim: 2, lower bound: -38.0363591, upper bound: 38.1205418
IS_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 232.59
Output dim: 2, lower bound: -38.1004078, upper bound: 38.2075008
IS_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 232.59
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
IS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 232.59
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524
IS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 232.59
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080522
IS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 232.59
Output dim: 2, lower bound: -38.1017021, upper bound: 38.2080524

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 163.81 + 7174.23 = 7338.04 seconds
